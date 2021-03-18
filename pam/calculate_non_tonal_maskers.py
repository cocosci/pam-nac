from utilities_n_libraries import *


def find_non_tonal_set(Flags, psd):
    """
    Description: given the psd and detected tonal maskers, find a set of non-tonal maskers
    Reference: ISO/IEC 11172-3:1993, Information technology Coding of moving pictures 
                and associated audio for digital storage media at up to about 1,5 Mbit/s
                Part 3: Audio, with the permission of ISO.
    """
    Non_tonal_list = np.zeros((len(CB) - 1, 2))
    for i in range(len(CB) - 1):
        # For each critical band, compute the power in non-tonal components
        power = MIN_POWER  # Partial sum
        weight = 0  # Used to compute the geometric mean of the critical band
        for k in range(int(TH[CB[i], INDEX]), int(TH[CB[i + 1], INDEX])):  # In each critical band
            # The index number for the non tonal component is the index nearest to the geometric mean 
            # of the critical band
            if Flags[k] == NOT_EXAMINED:
                power = 10 * ma.log10(10 ** (power / 10) + 10 ** (psd[k][0] / 10))
                weight = weight + 10 ** (psd[k][0] / 10) * (TH[Map[k], BARK] - TH[CB[i], BARK])
                Flags[k] = IRRELEVANT
        if power <= MIN_POWER:
            index = int(int(TH[CB[i], INDEX]))
        else:
            index = int(TH[CB[i], INDEX]) + np.round(
                weight / 10 ** (power / 10) * (int(TH[CB[i + 1], INDEX]) - int(TH[CB[i], INDEX])))
        if index < 0:
            index = 0
        if index > len(Flags) - 1:
            index = len(Flags) - 1
        index = int(index)
        if Flags[index] == TONAL:
            index = index + 1  # Two tonal components cannot be consecutive
        Non_tonal_list[i, INDEX] = int(index)
        Non_tonal_list[i, SPL] = power
        Flags[index] = NON_TONAL
    return Flags, Non_tonal_list
