from utilities_n_libraries import *


def individual_masking_thresholds(psd, Tonal_list, Non_tonal_list, TH, Map):
    """
    Description: for each of the valid masker, calculate the corresponding masking curve.
    """
    if Tonal_list.shape[0] == 0:
        LTt = []
    else:
        LTt = np.zeros((len(Tonal_list[:, 0]), len(TH[:, 0]))) + MIN_POWER
    LTn = np.zeros((len(Non_tonal_list[:, 0]), len(TH[:, 0]))) + MIN_POWER

    for i in range(len(TH[:, 0])):
        zi = TH[i, BARK]  # Critical band rate of the frequency considered

        if Tonal_list.shape[0] > 0:
            # For each tonal component
            for k in range(len(Tonal_list[:, 0])):
                j = int(Tonal_list[k, INDEX])
                zj = TH[Map[j], BARK]  # Critical band rate of the masker
                dz = zi - zj  # Distance in Bark to the masker
                if -3 <= dz < 8:
                    # Masking index
                    avtm = -1.525 - 0.275 * zj - 4.5
                    # Masking function
                    if -3 <= dz < -1:
                        vf = 17 * (dz + 1) - (0.4 * psd[j] + 6)
                    elif -1 <= dz < 0:
                        vf = (0.4 * psd[j] + 6) * dz
                    elif 0 <= dz < 1:
                        vf = -17 * dz
                    elif 1 <= dz < 8:
                        vf = - (dz - 1) * (17 - 0.15 * psd[j]) - 17
                    LTt[k, i] = Tonal_list[k, SPL] + avtm + vf

        # For each non-tonal component
        for k in range(len(Non_tonal_list[:, 0])):
            j = int(Non_tonal_list[k, INDEX])
            zj = TH[Map[j], BARK]  # Critical band rate of the masker
            dz = zi - zj  # Distance in Bark to the masker
            if -3 <= dz < 8:
                # Masking index
                avnm = -1.525 - 0.175 * zj - 0.5
                # Masking function
                if -3 <= dz < -1:
                    vf = 17 * (dz + 1) - (0.4 * psd[j] + 6)
                elif -1 <= dz < 0:
                    vf = (0.4 * psd[j] + 6) * dz
                elif 0 <= dz < 1:
                    vf = -17 * dz
                elif 1 <= dz < 8:
                    vf = - (dz - 1) * (17 - 0.15 * psd[j]) - 17
                LTn[k, i] = Non_tonal_list[k, SPL] + avnm + vf
    return LTt, LTn
