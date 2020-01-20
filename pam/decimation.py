from utilities_n_libraries import *


def decimation(Flags, Tonal_list, Non_tonal_list):
    """
    Description: remove tonal and non tonal candidates that are below the absolute hearing
                 threshold or half of a critical band from the adjacent component.
    Reference: Information technology -- Coding of moving pictures and associated
                audio for digital storage media at up to 1,5 Mbits/s -- Part3: audio.
                British standard. BSI, London. October 1993. Implementation of ISO/IEC
                11172-3:1993. BSI, London. First edition 1993-08-01.
    """
    DFlags = Flags

    DNon_tonal_list = np.empty((0, 2))
    for i in range(len(Non_tonal_list[:, 0])):
        k = int(Non_tonal_list[i, INDEX])
        if Non_tonal_list[i, SPL] < TH[Map[k], ATH]:
            DFlags[k] = IRRELEVANT
        else:
            DNon_tonal_list = np.concatenate((DNon_tonal_list, np.reshape(Non_tonal_list[i, :], (-1, 2))))

    DTonal_list = np.empty((0, 2))
    for i in range(len(Tonal_list[:, 0])):
        k = int(Tonal_list[i, INDEX])
        if Tonal_list[i, SPL] < TH[Map[k], ATH]:
            DFlags[k] = IRRELEVANT
        else:
            DTonal_list = np.concatenate((DTonal_list, np.reshape(Tonal_list[i, :], (-1, 2))))
    return DFlags, DTonal_list, DNon_tonal_list
