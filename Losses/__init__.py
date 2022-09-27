import torch
import torch.nn as nn


def get_dist_and_masks(embeddings, labels):

    n = embeddings.size(0)
    # L2 distance:
    dist = torch.norm(embeddings[:, None] - embeddings, dim=2, p=2)

    mask_pos = labels.expand(n, n).eq(labels.expand(n, n).t())
    mask_neg = ~mask_pos
    mask_pos[torch.eye(n).bool()] = 0

    return dist, mask_pos, mask_neg


# The base class for calculating loss and acc in main experiments:
class BatchHardTriplet(nn.Module):

    def __init__(self, margin):

        super(BatchHardTriplet, self).__init__()
        self.margin = margin

    def forward(self, dist, mask_pos, mask_neg, phase, sample_count_min):

        # Initialize matrix with same shape as dist containing -infinity:
        neg_infinity = torch.ones_like(dist) * float('-inf')

        if phase == 'train':

            # a = anchor, p = positive, n = negative:
            dist_ap = torch.where(mask_pos, dist, neg_infinity)
            # Use -dist so that the max operation can be used below to return the minimum an distance:
            dist_an = torch.where(mask_neg, -dist, neg_infinity)

            # Calculate max ap distances (hardest positive for each anchor):
            dist_ap_max = torch.max(dist_ap, dim=1)[0]

            # Find indices of valid ap pairs:
            indices_dist_ap_valid = torch.where(dist_ap_max != float('-inf'))[0]

            # Handle situation where there are no valid ap pairs (and thus no valid triplets) in the mini-batch:
            if len(indices_dist_ap_valid) == 0:

                loss = None
                acc = None

                print('No valid positives in batch')

            else:

                # Get valid ap pairs:
                dist_ap_max = dist_ap_max[indices_dist_ap_valid].unsqueeze(1)

                # Get min an distances from anchors with at least one valid positive. Then, multiply by -1 at the
                # end to reverse the -dist operation above in dist_an definition:
                dist_an_min = torch.max(dist_an, dim=1)[0][indices_dist_ap_valid] * -1
                dist_an_min = dist_an_min.unsqueeze(1)

                loss = torch.clamp(dist_ap_max - dist_an_min + self.margin, min=0.).mean()

                with torch.no_grad():

                    acc = (dist_an_min > dist_ap_max).float().mean().item()

        else:

            dist_ap = torch.where(mask_pos, dist, neg_infinity)
            dist_an = torch.where(mask_neg, dist, neg_infinity)

            # Reduce ap and an distance matrices to contain only session two samples as anchors (row-wise) and
            # session one samples as positives or negatives (column-wise). These matrices are organised such that
            # there are sample_count_min samples from session one and then sample_count_min samples from session two
            # for each ID. As such, they are split into subsets of sample_count_min, and then every second subset is
            # taken to get all samples from a given session ([1::2] is a slice containing all session two samples,
            # [::2] is a slice containing all session one samples):
            chunk = torch.cat(torch.split(dist_ap, int(sample_count_min))[1::2], dim=0)
            dist_ap_red = torch.cat(torch.split(chunk, int(sample_count_min), dim=1)[::2], dim=1)  # red means reduced

            chunk = torch.cat(torch.split(dist_an, int(sample_count_min))[1::2], dim=0)
            dist_an_red = torch.cat(torch.split(chunk, int(sample_count_min), dim=1)[::2], dim=1)

            # Remove data from original distance matrices to save memory (not used again):
            dist_ap = None
            dist_an = None

            # Get indices of valid positives and negatives for each anchor:
            indices_dist_ap_valid = torch.nonzero(dist_ap_red != float('-inf'), as_tuple=False)
            indices_dist_an_valid = torch.nonzero(dist_an_red != float('-inf'), as_tuple=False)

            # Get valid ap and an pairs:
            dist_ap_red_valid = torch.stack(torch.split(dist_ap_red[indices_dist_ap_valid[:, 0],
                                                                    indices_dist_ap_valid[:, 1]],
                                                        int(sample_count_min)))
            dist_an_red_valid = torch.stack(torch.split(dist_an_red[indices_dist_an_valid[:, 0],
                                                                    indices_dist_an_valid[:, 1]],
                                                        int(dist_an_red.size(1) - sample_count_min)))

            # Remove data from previous distance matrices to save memory (not used again):
            dist_ap_red = None
            dist_an_red = None

            # Split the an distance matrix by ID (there are now sample_count_min samples per negative ID) and then take
            # the min across the IDs:
            dist_an_min = torch.min(torch.stack(torch.split(dist_an_red_valid, int(sample_count_min), dim=1)),
                                    dim=0, keepdim=False)[0]

            loss = torch.clamp(dist_ap_red_valid - dist_an_min + self.margin, min=0.).mean()

            with torch.no_grad():

                acc = (dist_an_min > dist_ap_red_valid).float().mean().item()

        return loss, acc


# The class used for reducing validation and test set ap and an distance matrices to contain a pre-defined no. of
# samples per ID (n_samples_per_ID) to compare with each anchor. This class was used to validate the performance
# evaluation setting (see Supp Material - Section I). Note that it is the same as the BatchHardTriplet class above
# until after the 'dist_an_red = None' line:
class BatchHardTripletSuppPt1(nn.Module):

    def __init__(self, margin):

        super(BatchHardTripletSuppPt1, self).__init__()
        self.margin = margin

    def forward(self, dist, mask_pos, mask_neg, phase, sample_count_min, n_samples_per_ID):

        # Initialize matrix with same shape as dist containing -infinity:
        neg_infinity = torch.ones_like(dist) * float('-inf')

        if phase == 'train':

            # a = anchor, p = positive, n = negative:
            dist_ap = torch.where(mask_pos, dist, neg_infinity)
            # Use -dist so that the max operation can be used below to return the minimum an distance:
            dist_an = torch.where(mask_neg, -dist, neg_infinity)

            # Calculate max ap distances (hardest positive for each anchor):
            dist_ap_max = torch.max(dist_ap, dim=1)[0]

            # Find indices of valid ap pairs:
            indices_dist_ap_valid = torch.where(dist_ap_max != float('-inf'))[0]

            # Handle situation where there are no valid ap pairs (and thus no valid triplets) in the mini-batch:
            if len(indices_dist_ap_valid) == 0:

                loss = None
                acc = None

                print('No valid positives in batch')

            else:

                # Get valid ap pairs:
                dist_ap_max = dist_ap_max[indices_dist_ap_valid].unsqueeze(1)

                # Get min an distances from anchors with at least one valid positive. Then, multiply by -1 at the
                # end to reverse the -dist operation above in dist_an definition:
                dist_an_min = torch.max(dist_an, dim=1)[0][indices_dist_ap_valid] * -1
                dist_an_min = dist_an_min.unsqueeze(1)

                loss = torch.clamp(dist_ap_max - dist_an_min + self.margin, min=0.).mean()

                with torch.no_grad():

                    acc = (dist_an_min > dist_ap_max).float().mean().item()

        else:

            dist_ap = torch.where(mask_pos, dist, neg_infinity)
            dist_an = torch.where(mask_neg, dist, neg_infinity)

            # Reduce ap and an distance matrices to contain only session two samples as anchors (row-wise) and
            # session one samples as positives or negatives (column-wise). These matrices are organised such that
            # there are sample_count_min samples from session one and then sample_count_min samples from session two
            # for each ID. As such, they are split into subsets of sample_count_min, and then every second subset is
            # taken to get all samples from a given session ([1::2] is a slice containing all session two samples,
            # [::2] is a slice containing all session one samples):
            chunk = torch.cat(torch.split(dist_ap, int(sample_count_min))[1::2], dim=0)
            dist_ap_red = torch.cat(torch.split(chunk, int(sample_count_min), dim=1)[::2], dim=1)  # red means reduced

            chunk = torch.cat(torch.split(dist_an, int(sample_count_min))[1::2], dim=0)
            dist_an_red = torch.cat(torch.split(chunk, int(sample_count_min), dim=1)[::2], dim=1)

            # Remove data from original distance matrices to save memory (not used again):
            dist_ap = None
            dist_an = None

            # Get indices of valid positives and negatives for each anchor:
            indices_dist_ap_valid = torch.nonzero(dist_ap_red != float('-inf'), as_tuple=False)
            indices_dist_an_valid = torch.nonzero(dist_an_red != float('-inf'), as_tuple=False)

            # Get valid ap and an pairs:
            dist_ap_red_valid = torch.stack(torch.split(dist_ap_red[indices_dist_ap_valid[:, 0],
                                                                    indices_dist_ap_valid[:, 1]],
                                                        int(sample_count_min)))
            dist_an_red_valid = torch.stack(torch.split(dist_an_red[indices_dist_an_valid[:, 0],
                                                                    indices_dist_an_valid[:, 1]],
                                                        int(dist_an_red.size(1) - sample_count_min)))

            # Remove data from previous distance matrices to save memory (not used again):
            dist_ap_red = None
            dist_an_red = None

            # Further reduce ap distance matrix to contain only n_samples_per_ID to compare with each anchor.
            dist_ap_red_2 = torch.min(dist_ap_red_valid[:, :n_samples_per_ID], dim=1, keepdim=False)[0]
            # Further reduce an distance matrix to contain only n_samples_per_ID to compare with each anchor.
            # To get the global min an distance for each anchor, first take the min across the sample dim (dim=2),
            # then take the min across the ID dim (dim=0):
            dist_an_red_2 = torch.min(torch.min(torch.stack(torch.split(dist_an_red_valid, int(sample_count_min),
                                                                        dim=1))[:, :, :n_samples_per_ID], dim=2,
                                                keepdim=False)[0], dim=0, keepdim=False)[0]

            loss = torch.clamp(dist_ap_red_2 - dist_an_red_2 + self.margin, min=0.).mean()

            with torch.no_grad():

                acc = (dist_an_red_2 > dist_ap_red_2).float().mean().item()

        return loss, acc


# The class used when not reducing the no. of samples per ID to compare with each anchor in validation and test
# sets. This is the 'All' condition in Supp Material - Section I:
class BatchHardTripletSuppPt2(nn.Module):

    def __init__(self, margin):

        super(BatchHardTripletSuppPt2, self).__init__()
        self.margin = margin

    def forward(self, dist, mask_pos, mask_neg, phase, indices_s1, indices_s2):

        # Initialize matrix with same shape as dist containing -infinity:
        neg_infinity = torch.ones_like(dist) * float('-inf')

        if phase == 'train':

            # a = anchor, p = positive, n = negative:
            dist_ap = torch.where(mask_pos, dist, neg_infinity)
            # Use -dist so that the max operation can be used below to return the minimum an distance:
            dist_an = torch.where(mask_neg, -dist, neg_infinity)

            # Calculate max ap distances (hardest positive for each anchor):
            dist_ap_max = torch.max(dist_ap, dim=1)[0]

            # Find indices of valid ap pairs:
            indices_dist_ap_valid = torch.where(dist_ap_max != float('-inf'))[0]

            # Handle situation where there are no valid ap pairs (and thus no valid triplets) in the mini-batch:
            if len(indices_dist_ap_valid) == 0:

                loss = None
                acc = None

                print('No valid positives in batch')

            else:

                # Get valid ap pairs:
                dist_ap_max = dist_ap_max[indices_dist_ap_valid].unsqueeze(1)

                # Get min an distances from anchors with at least one valid positive. Then, multiply by -1 at the
                # end to reverse the -dist operation above in dist_an definition:
                dist_an_min = torch.max(dist_an, dim=1)[0][indices_dist_ap_valid] * -1
                dist_an_min = dist_an_min.unsqueeze(1)

                loss = torch.clamp(dist_ap_max - dist_an_min + self.margin, min=0.).mean()

                with torch.no_grad():

                    acc = (dist_an_min > dist_ap_max).float().mean().item()

        else:

            dist_ap = torch.where(mask_pos, dist, neg_infinity)
            dist_an = torch.where(mask_neg, dist, neg_infinity)

            # Reduce ap and an distance matrices to contain only session two samples as anchors (row-wise) and
            # session one samples as positives or negatives (column-wise):
            dist_ap_red = dist_ap[indices_s2, :][:, indices_s1]  # red means reduced
            dist_an_red = dist_an[indices_s2, :][:, indices_s1]

            # Remove data from original distance matrices to save memory (not used again):
            dist_ap = None
            dist_an = None

            # Get indices of valid positives and negatives for each anchor:
            indices_dist_ap_valid = torch.nonzero(dist_ap_red != float('-inf'), as_tuple=False)
            indices_dist_an_valid = torch.nonzero(dist_an_red != float('-inf'), as_tuple=False)

            # Get the no. of valid positives and negatives for each ID:
            counts_p = list(torch.unique(indices_dist_ap_valid[:, 0], return_counts=True)[1].cpu().numpy())
            counts_n = list(torch.unique(indices_dist_an_valid[:, 0], return_counts=True)[1].cpu().numpy())

            # Get valid ap and an pairs and split the distance matrices by ID:
            dist_ap_red_valid = torch.split(dist_ap_red[indices_dist_ap_valid[:, 0],
                                                        indices_dist_ap_valid[:, 1]], counts_p)
            dist_an_red_valid = torch.split(dist_an_red[indices_dist_an_valid[:, 0],
                                                        indices_dist_an_valid[:, 1]], counts_n)

            # Remove data from previous distance matrices to save memory (not used again):
            dist_ap_red = None
            dist_an_red = None

            # Get the min distance across all valid ap distances for each anchor:
            dist_ap_min = torch.tensor([torch.min(item) for item in dist_ap_red_valid])
            # Get the min distance across all valid an distances for each anchor:
            dist_an_min = torch.tensor([torch.min(item) for item in dist_an_red_valid])

            loss = torch.clamp(dist_ap_min - dist_an_min + self.margin, min=0.).mean()

            with torch.no_grad():

                acc = (dist_an_min > dist_ap_min).float().mean().item()

        return loss, acc
