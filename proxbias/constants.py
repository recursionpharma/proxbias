from importlib import resources
from typing import List

# Source of truth:
# https://docs.google.com/spreadsheets/d/1_ksX-JQ4czJ18XLxYeWJmpXCCLjneNBz06U5imBMXdo/edit#gid=1916322961
# fmt: off
RXRX3_GENE_EXCLUSIONS = \
    ['ABCA1', 'ABHD5', 'ACVR2A', 'ADAM10', 'ADAM15', 'ADAR', 'AKIRIN1', 'ANXA1',
     'ANXA6', 'AP1S3', 'APC', 'APOD', 'ARCN1', 'ARF6', 'ARID1A', 'ARID2', 'ASCL2',
     'ATF4', 'ATF7IP', 'ATG101', 'ATG13', 'ATG14', 'ATG2B', 'ATG7', 'ATG9A',
     'ATP23', 'ATP5IF1', 'ATP5PO', 'ATXN2', 'AXIN1', 'BAX', 'BCL9', 'BEAN1',
     'BECN1', 'BIRC2', 'CALR', 'CAPN3', 'CAV1', 'CCN3', 'CCND1', 'CCNDBP1', 'CCNE1',
     'CD9', 'CDK1', 'CDK12', 'CDX2', 'CEPT1', 'CFAP410', 'COL1A1', 'COL9A1', 'COP1',
     'CTNNB1', 'DAG1', 'DDR2', 'DKK1', 'DNAJB12', 'DOCK5', 'DUOX1', 'DUSP1', 'E2F3',
     'ECM1', 'EFNB1', 'EIF4E2', 'ELAVL1', 'EMC2', 'EPHB1', 'ERBB4', 'ERG', 'ERN1',
     'ETNK2', 'EZH2', 'FABP3', 'FANCM', 'FASTKD1', 'FBXW7', 'FN1', 'FOSL1', 'FOXF2',
     'FOXO1', 'FOXP1', 'FUS', 'FZD7', 'FZD9', 'GAP43', 'GDF9', 'GLIS2', 'GLS',
     'GLS2', 'GPR143', 'GPR3', 'GPRC5A', 'GPX4', 'GREM1', 'HOXB9', 'IGF1', 'IGFBP1',
     'IL17RD', 'ILDR1', 'IPP', 'IRX6', 'ITGAV', 'JAK1', 'KAT6A', 'KCTD5', 'KDM5A',
     'KDM5C', 'KEAP1', 'KIF5A', 'KIRREL1', 'KLF4', 'KLF5', 'KLK6', 'KPNA1', 'LAMB3',
     'LARGE1', 'LATS2', 'LBX2', 'LGR5', 'LGR6', 'LMNB2', 'LOXL1', 'LPIN1', 'MAP3K7',
     'MAP4K4', 'MBNL1', 'MBTPS1', 'MCRS1', 'MCU', 'MDK', 'MOBP', 'MSTN', 'MTPN',
     'MYC', 'MYH9', 'MYOM3', 'NFE2L2', 'NINJ2', 'NOTCH1', 'NOTUM', 'OLFM4', 'PAX8',
     'PDPN', 'PIKFYVE', 'PKN2', 'PLAUR', 'PNPLA2', 'PODXL', 'POLDIP3', 'POLG',
     'POLG2', 'POSTN', 'PPFIBP2', 'PPP1R3B', 'PPP2R1A', 'PPP3CA', 'PPP6C', 'PRKAA1',
     'PRKCSH', 'PTGDR', 'PTGFRN', 'PTK2', 'PTN', 'PTPN12', 'PTPN14', 'RB1CC1',
     'RBM39', 'RFTN1', 'RFX5', 'RNF31', 'RNF43', 'RSPO2', 'RSPO3', 'RUNX1',
     'S100A11', 'S100A6', 'SAR1B', 'SARM1', 'SCFD1', 'SERPINH1', 'SETD2', 'SETDB1',
     'SHOC2', 'SLC5A11', 'SLC8A3', 'SMAD4', 'SMARCA1', 'SMARCA2', 'SMARCA4',
     'SMARCB1', 'SMPD1', 'SNAI1', 'SOCS3', 'SOD1', 'SOX15', 'SOX2', 'SOX9',
     'SPTLC1', 'SRSF1', 'STING1', 'STK11', 'STK3', 'SUSD1', 'SYNM', 'TARBP2',
     'TARDBP', 'TBC1D22A', 'TBK1', 'TCF7L2', 'TEAD1', 'TEC', 'TERF2', 'TGFB1',
     'THBS1', 'TIAM1', 'TIMELESS', 'TM4SF4', 'TMEM97', 'TRAF2', 'TRAF3', 'TRIP13',
     'TSPAN14', 'TXK', 'TYRP1', 'UBE2C', 'ULK1', 'UMODL1', 'UNC13A', 'USP20',
     'USP7', 'VWF', 'WNT10B', 'WNT7A', 'XBP1', 'YAP1', 'ZC3H12A', 'ZEB1', 'ZNF107',
     'ZNRF3']
# fmt: on

OTHER_GENE_EXCLUSIONS: List[str] = []

GENE_FILTER_LIST = frozenset(RXRX3_GENE_EXCLUSIONS + OTHER_GENE_EXCLUSIONS)


VALID_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]

DATA_DIR = resources.files("proxbias").joinpath("data")  # type:ignore[attr-defined]
