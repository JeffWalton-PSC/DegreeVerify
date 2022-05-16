import os
import sys
import argparse as ap
import datetime as dt
import numpy as np
import pandas as pd
from loguru import logger


import powercampus as pc


logger.remove()
logger.add(sys.stdout, level="WARNING")
logger.add(sys.stderr, level="WARNING")
logger.add(
    "logs/DegreeVerify.log",
    rotation="monthly",
    format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {name} | {message}",
    level="INFO",
)


def people_data() -> pd.DataFrame:
    """
    returns data from PEOPLE table
    """
    df_p = pc.select("PEOPLE",
                fields=['PEOPLE_CODE_ID', 'GOVERNMENT_ID', 'FIRST_NAME', 'MIDDLE_NAME', 'LAST_NAME',
                        'SUFFIX', 'BIRTH_DATE', 'RELEASE_INFO'
                        ],
                where="BIRTH_DATE > '1800-01-01' and BIRTH_DATE < '2100-12-31' "
                )
    df_p['MIDDLE_NAME'] = df_p['MIDDLE_NAME'].str.replace('.','', regex=False).fillna(' ')
    df_p['SUFFIX'] = df_p['SUFFIX'].str.replace('.','', regex=False).fillna(' ')
    df_p['BIRTH_DATE'] = df_p['BIRTH_DATE'].fillna('1900-01-01 00:00:00')
    df_p['BIRTH_DATE'] = df_p['BIRTH_DATE'].dt.strftime('%Y%m%d')
    df_p['BIRTH_DATE'] = df_p['BIRTH_DATE'].str.replace('19000101', ' ')
    return df_p


def academic_cal_data() -> pd.DataFrame:
    """
    returns start and end dates for terms from ACADEMICCALENDAR
    """
    df_acal = pc.select("ACADEMICCALENDAR",
                fields=['ACADEMIC_YEAR', 'ACADEMIC_TERM', 'ACADEMIC_SESSION', 'START_DATE', 'END_DATE', 'FINAL_END_DATE'],
                where="ACADEMIC_YEAR > 1999 "
                )
    df_acal = ( df_acal.groupby(['ACADEMIC_YEAR', 'ACADEMIC_TERM']).agg(
            {'START_DATE': ['min'],
            'END_DATE': ['max'],
            'FINAL_END_DATE': ['max'],
            }
        )
        .reset_index()
        .droplevel(1, axis=1)
    )
    df_acal['END_DATE'] = df_acal[['END_DATE', 'FINAL_END_DATE']].max(axis=1)
    df_acal = df_acal.drop(columns=['FINAL_END_DATE'])
    return df_acal


def academic_data() -> pd.DataFrame:
    """
    returns student's start and end dates of attendance from ACADEMIC table
    """
    df_a = pc.select("ACADEMIC",
                fields=['PEOPLE_CODE_ID', 'ACADEMIC_YEAR', 'ACADEMIC_TERM', 'ACADEMIC_SESSION',
                        'PROGRAM', 'DEGREE', 'CURRICULUM', 'COLLEGE', 
                        'ADMIT_YEAR', 'ADMIT_TERM', 'ADMIT_DATE'],
                where="ACADEMIC_SESSION = '' and PRIMARY_FLAG = 'Y' and CREDITS > 0 " + 
                    "and ADMIT_DATE IS NOT NULL and ADMIT_DATE > '1899-01-01' and ADMIT_DATE < '2100-01-01'"
                )
    df_a = pc.select("ACADEMIC",
                fields=['PEOPLE_CODE_ID', 'ACADEMIC_YEAR', 'ACADEMIC_TERM', 'ACADEMIC_SESSION',
                        'ENROLL_SEPARATION', ],
                where="ACADEMIC_SESSION = '' and PRIMARY_FLAG = 'Y' and CREDITS > 0 " + 
                    "and ACADEMIC_YEAR > 1999 and ACADEMIC_TERM NOT IN ('Transfer', 'JTERM') and ENROLL_SEPARATION = 'ENRL' "
                )
    df_a = df_a.drop(
                    columns=[ 'ACADEMIC_SESSION', 'ENROLL_SEPARATION' ],
                )
    df_a = pc.add_col_yearterm_sort(df_a)

    df_acal = academic_cal_data()
    df_start = ( df_a
                .sort_values(['PEOPLE_CODE_ID', 'yearterm_sort'])
                .drop_duplicates(subset=['PEOPLE_CODE_ID'], keep='first')
    )
    df_start = ( df_start.merge(
                    df_acal[['ACADEMIC_YEAR', 'ACADEMIC_TERM', 'START_DATE']],
                    how='left',
                    on=[ 'ACADEMIC_YEAR', 'ACADEMIC_TERM']
                    )
                    .drop(
                        columns=[ 'ACADEMIC_YEAR', 'ACADEMIC_TERM', 'yearterm_sort' ],
                    )
            )
    df_end = ( df_a
                .sort_values(['PEOPLE_CODE_ID', 'yearterm_sort'])
                .drop_duplicates(subset=['PEOPLE_CODE_ID'], keep='last')
    )
    df_end = ( df_end.merge(
                    df_acal[['ACADEMIC_YEAR', 'ACADEMIC_TERM', 'END_DATE']],
                    how='left',
                    on=[ 'ACADEMIC_YEAR', 'ACADEMIC_TERM']
                    )
                    .drop(
                        columns=[ 'ACADEMIC_YEAR', 'ACADEMIC_TERM', 'yearterm_sort' ],
                    )
            )

    df_a = ( df_a[['PEOPLE_CODE_ID']].groupby(['PEOPLE_CODE_ID']).first()
            .reset_index()
    )
    df_a = ( df_a.merge(
                    df_start,
                    how='left',
                    on='PEOPLE_CODE_ID'
                )
                .merge(
                    df_end,
                    how='left',
                    on='PEOPLE_CODE_ID'
                )
        )
    df_a['START_DATE'] = df_a['START_DATE'].dt.strftime('%Y%m%d')
    df_a['END_DATE'] = df_a['END_DATE'].dt.strftime('%Y%m%d')
    return df_a


def transcript_degree_data(start_date:np.datetime64, end_date:np.datetime64) -> pd.DataFrame:
    """
    returns students that received degrees from TRANSCRIPTDEGREE table
    """
    df_td = pc.select("TRANSCRIPTDEGREE",
                fields=['PEOPLE_CODE_ID', 'PROGRAM', 'DEGREE', 'CURRICULUM', 'FORMAL_TITLE', 'GRADUATION_DATE'],
                where=f"GRADUATION_DATE IS NOT NULL and GRADUATION_DATE >= '{start_date}' and GRADUATION_DATE <= '{end_date}' " +
                    "and DEGREE <> 'NOND' "
                )
    df_td = ( df_td
                .sort_values(['PEOPLE_CODE_ID', 'PROGRAM', 'DEGREE', 'CURRICULUM', 'GRADUATION_DATE'])
                .drop_duplicates(subset=['PEOPLE_CODE_ID', 'PROGRAM', 'DEGREE', 'CURRICULUM', 'GRADUATION_DATE'], keep='last')
                .drop(
                    columns=['PROGRAM'],
                )
    )
    df_td['GRADUATION_DATE'] = df_td['GRADUATION_DATE'].dt.strftime('%Y%m%d')
    return df_td


def degree_mapping() -> pd.DataFrame:
    """
    returns CIP code for each CURRICULUM from DegreeMAppingNsc table
    """
    df_dm = pc.select("DegreeMappingNsc",
                fields=['AcademicYear', 'AcademicTerm', 'Degree', 'Curriculum', 'CipCode', 'CipYear']
                )
    df_dm = ( df_dm
            .sort_values(['AcademicYear', 'Curriculum', ])
            .drop_duplicates(subset=['Curriculum', ], keep='last')
            .drop( 
                columns=['AcademicYear', 'AcademicTerm', 'CipYear'],
            )
            )
    df_dm['CipCode']=df_dm['CipCode'].str.replace('.','', regex=False)
    return df_dm


def code_degree() -> pd.DataFrame:
    """
    returns long description for each degree code
    """
    df_code_degree = pc.select("CODE_DEGREE",
                fields=['CODE_VALUE_KEY', 'LONG_DESC']
                )
    df_code_degree = ( df_code_degree
            .sort_values(['CODE_VALUE_KEY', ])
            )
    return df_code_degree


def transcript_honors() -> pd.DataFrame:
    """
    returns dataframe of students with HONORS
    """
    df_th = pc.select("TRANSCRIPTHONORS",
                fields=['PEOPLE_CODE_ID', 'PROGRAM', 'DEGREE', 'CURRICULUM', 'HONORS']
                )
    df_th = ( df_th
                .sort_values(['PEOPLE_CODE_ID', 'PROGRAM', 'DEGREE', 'CURRICULUM', 'HONORS'])
                .drop_duplicates(subset=['PEOPLE_CODE_ID', 'PROGRAM', 'DEGREE', 'CURRICULUM', 'HONORS'], keep='last')
                .drop(
                    columns=['PROGRAM'],
                )
    )
    return df_th


def people_formername(people_df: pd.DataFrame) -> pd.DataFrame:
    """
    returns a dataframe of all people with former names appended
    """
    df_pfn = pc.select("PEOPLEFORMERNAME",
                fields=['PEOPLE_CODE_ID', 'FIRST_NAME', 'MIDDLE_NAME', 'LAST_NAME', 'NAME_CHANGE_DATE'],
                )
    df_pfn = ( df_pfn
                .sort_values(['PEOPLE_CODE_ID', 'NAME_CHANGE_DATE'])
                .drop_duplicates(subset=['PEOPLE_CODE_ID'], keep='last')
                .rename(
                    columns={'FIRST_NAME': 'Previous First Name',
                            'LAST_NAME': 'Previous Last Name'
                    }
                )
                .drop(
                    columns=['MIDDLE_NAME', 'NAME_CHANGE_DATE'],
                )
    )

    df_p = people_df.merge(
        df_pfn,
        how='left',
        on='PEOPLE_CODE_ID'
    )
    df_p[['Previous First Name', 'Previous Last Name']] = df_p[['Previous First Name', 'Previous Last Name']].fillna(' ')
    return df_p


def minor_data(df_td: pd.DataFrame) -> pd.DataFrame:
    """
    returns a dataframe of minors
    """
    df_minor = ( df_td.loc[(df_td['DEGREE']=='MINOR'),:]
                .sort_values(['PEOPLE_CODE_ID', 'GRADUATION_DATE', ])
                .drop_duplicates(['PEOPLE_CODE_ID', 'GRADUATION_DATE', 'DEGREE', 'CURRICULUM'], keep="last")
            )
    minor_rank = df_minor.groupby(["PEOPLE_CODE_ID"])["GRADUATION_DATE"]
    df_minor["rank"] = minor_rank.rank(method="first").astype(int)
    df_minor = df_minor.loc[(df_minor['rank']<=4)]
    df_minor["col"] = "Minor Course of Study " + df_minor["rank"].astype(str)
    df_minor['FORMAL_TITLE'] = df_minor['FORMAL_TITLE'].str.replace(' Minor', '').replace('Minor', '')
    minor = df_minor.pivot(index=["PEOPLE_CODE_ID", 'GRADUATION_DATE'], columns="col", values=["FORMAL_TITLE"]).fillna(' ')
    minor = minor.droplevel(0, axis=1).reset_index()
    return minor


def stoplist_data() -> pd.DataFrame:
    """
    returns dataframe of students with outstanding financial obligations 
    """
    df_sl = pc.select("STOPLIST",
                fields=['PEOPLE_CODE_ID', 'STOP_REASON', 'STOP_DATE'],
                where="STOP_REASON in ('BURS', 'COLL', 'STAC') and CLEARED='N' "
                )
    df_sl = ( df_sl
                .sort_values(['PEOPLE_CODE_ID', 'STOP_DATE'])
                .drop_duplicates(subset=['PEOPLE_CODE_ID'], keep='last')
                .drop(
                    columns=['STOP_DATE'],
                )
    )
    return df_sl


def create_DV_df(start_date:np.datetime64, end_date:np.datetime64) -> pd.DataFrame:
    """
    create dataframe for Degree Verify data
    """
    df_td = transcript_degree_data(start_date, end_date)
    logger.info(f"{df_td.shape=}")

    df_a = academic_data()
    logger.info(f"{df_a.shape=}")

    df_p = people_data()
    logger.info(f"{df_p.shape=}")

    df_p = people_formername(df_p)
    logger.info(f"{df_p.shape=}")

    df_minor = minor_data(df_td)
    logger.info(f"{df_minor.shape=}")

    df_th = transcript_honors()
    logger.info(f"{df_th.shape=}")

    df_dm = degree_mapping()
    logger.info(f"{df_dm.shape=}")

    df_code_degree = code_degree()
    logger.info(f"{df_code_degree.shape=}")

    df_stoplist = stoplist_data()
    logger.info(f"{df_stoplist.shape=}")

    # remove minors
    df_td = df_td.loc[(df_td['DEGREE']!='MINOR'),:]

    df = ( df_td.merge(
                    df_a,
                    how='left',
                    on='PEOPLE_CODE_ID'
                )
                .merge(
                    df_p,
                    how='left',
                    on='PEOPLE_CODE_ID'
                )
                .merge(
                    df_minor,
                    how='left',
                    on=['PEOPLE_CODE_ID', 'GRADUATION_DATE']
                )
                .merge(
                    df_th,
                    how='left',
                    on=['PEOPLE_CODE_ID', 'DEGREE', 'CURRICULUM']
                )
                .merge(
                    df_dm,
                    how='left',
                    left_on=['DEGREE', 'CURRICULUM'], 
                    right_on=['Degree', 'Curriculum'], 
                )
                .merge(
                    df_code_degree,
                    how='left',
                    left_on=['DEGREE'], 
                    right_on=['CODE_VALUE_KEY'], 
                )
                .merge(
                    df_stoplist,
                    how='left',
                    on=['PEOPLE_CODE_ID'], 
                )
    )

    df['Record Type']='DD1'

    df.loc[df['DEGREE'].isin(['BS','BA','BPS']), 'Degree Level Indicator'] = "B"
    df.loc[df['DEGREE'].isin(['AS','AAS','AA','AOS']), 'Degree Level Indicator'] = "A"
    df.loc[df['DEGREE'].isin(['MS', 'MPS']), 'Degree Level Indicator'] = "M"
    df.loc[df['DEGREE'].isin(['CERTIF']), 'Degree Level Indicator'] = "C"
    df.loc[df['DEGREE'].isin(['GCERT']), 'Degree Level Indicator'] = "T"
    df.loc[df['HONORS'].isin(['CUM']), 'Academic Honors'] = "Cum Laude"
    df.loc[df['HONORS'].isin(['MAGNA']), 'Academic Honors'] = "Magna Cum Laude"
    df.loc[df['HONORS'].isin(['SUMMA']), 'Academic Honors'] = "Summa Cum Laude"
    df.loc[df['DEGREE'].isin(['CERTIF']), 'Certificate Type'] = "2"

    # FERPA Block
    df['FERPA Block'] = "N"
    df.loc[df['RELEASE_INFO'].isin(['NORL']), 'FERPA Block'] = "Y"

    # School Financial Block
    df['School Financial Block'] = "N"
    df.loc[df['STOP_REASON'].isin(['BURS', 'COLL', 'STAC']), 'School Financial Block'] = "Y"

    df = df.rename(
        columns={
            'GOVERNMENT_ID': 'Student SSN',
            'FIRST_NAME': 'First Name',
            'MIDDLE_NAME': 'Middle Name',
            'LAST_NAME': 'Last Name',
            'SUFFIX': 'Name Suffix',
            'BIRTH_DATE': 'Date of Birth',
            'PEOPLE_CODE_ID': 'College Student ID',
            'GRADUATION_DATE': 'Date Degree, Credential, or Certificate Awarded',
            'LONG_DESC': 'Degree, Certificate, or Credential Title',
            'FORMAL_TITLE': 'Major Course of Study 1',
            'START_DATE': 'Attendance From Date',
            'END_DATE': 'Attendance To Date',
            'CipCode': 'NCES CIP Code for Major 1',
        }
    )

    unused_fields = [
        'School/College/Division Awarding Degree',
        'Joint Institution/College/School/Division Name',
        'Major Course of Study 2',
        'Major Course of Study 3',
        'Major Course of Study 4',
        'Minor Course of Study 1',
        'Minor Course of Study 2',
        'Minor Course of Study 3',
        'Minor Course of Study 4',
        'Major Option 1',
        'Major Option 2',
        'Major Concentration 1',
        'Major Concentration 2',
        'Major Concentration 3',
        'NCES CIP Code for Major 1',
        'NCES CIP Code for Major 2',
        'NCES CIP Code for Major 3',
        'NCES CIP Code for Major 4',
        'NCES CIP Code for Minor 1',
        'NCES CIP Code for Minor 2',
        'NCES CIP Code for Minor 3',
        'NCES CIP Code for Minor 4',
        'Honors Program',
        'Other Honors',
        'Name of Institution Granting Degree',
        'Reverse Transfer Flag',
        'Certificate Type',
        'Filler01',
        'Filler02',
        'Filler03',
        'Filler04',
        'Filler05',
        'Filler06',
        'Filler07',
        'Filler08',
        'Filler09',
        'Filler10',
        'Filler11',
        'Filler12',
    ]
    for f in unused_fields:
        if f not in df.columns:
            df[f] = ' '

    fill_list = [
        'Minor Course of Study 1',
        'Minor Course of Study 2',
        'Minor Course of Study 3',
        'Minor Course of Study 4',
        'Academic Honors',
        'Certificate Type',
        'NCES CIP Code for Major 1',
        'Attendance From Date',
        'Attendance To Date',
    ]
    df[fill_list] = df[fill_list].fillna(' ')

    # missing SSN
    df = df.loc[~(df['Student SSN'].isna()),:]
    # apply "NO SSN" label
    # foreign students: '000', '888', '999'
    df.loc[(df['Student SSN'].str.startswith('000')), 'Student SSN'] = "NO SSN"
    df.loc[(df['Student SSN'].str.startswith('888')), 'Student SSN'] = "NO SSN"
    df.loc[(df['Student SSN'].str.startswith('999')), 'Student SSN'] = "NO SSN"

    # fill missing 'First Name'
    df.loc[(df['First Name'].isna()), 'First Name'] = "NFN"

    df = df.sort_values(['Student SSN', 'Date Degree, Credential, or Certificate Awarded'])

    logger.info(f"{df.shape=}")
    # logger.info(f"{df.df.head(1)}")

    return df


def write_fw(fpath:str, f_mode:str, df:pd.DataFrame, specs:dict) -> int:
    """
    Write dataframe to fixed width column format with the given column specs
    
    Arguments:
        fpath: output file path
        f_mode: the file write mode 'a'=append, 'w'=write
        df: input dataframe
        specs: dictionary of columns to ouput, keys= column names, values=[string length, start position, end position]
    
    Return value:
        0: inconsistent field specifications
        >0: lines written
    """
    with open(fpath, f_mode) as f:
        # template = "1234567890"*384
        # f.write(template + "\n")
        lw = 0
        for index, row in df.iterrows():
            line=""
            for c, v in specs.items():
                logger.debug(f"{c=}")
                if "Filler" in c:
                    s = " "
                else:
                    s = row[c]
                logger.debug(f"{s=} {type(s)=}")
                max_len, start, end = v
                # check field start, end and max_len are consistent
                if (start + (max_len - 1)) != end:
                    logger.error(f"{c}: field specs are not consistent!")
                    return 0
                if len(s) > max_len:
                    s = s[:max_len]
                elif len(s) < max_len:
                    s = s.ljust(max_len)
                line += s
            
            f.write(line)
            f.write("\n")
            lw += 1
    
    return lw


def write_DV_header(out_fn: str, transmission_date:str, degree_period:str) -> int:
    """
    defines DegreeVerify header columns
    writes fixed-width column file

    Arguments:
    out_fn: output file name
    transmission_date: a string containing the date transmission was reported (YYYYMMDD)
    degree_period: a string containing a description of the time period being reported

    Return value:
    error code from write_fw()
    """

    # column definitions
    #   'column_name': [column_width, column_start_position, column_end_position]
    dv_hdr_def = {
        'Record Type':          [    3,   1,    3],
        'School Code':          [    6,   4,    9],
        'Branch Code':          [    2,  10,   11],
        'Official School Name': [   80,  12,   91],
        'Filler01':             [   15,  92,  106],
        'Standard Report Flag': [    1, 107,  107],
        'Transmission Date':    [    8, 108,  115],
        'Degree Period':        [   80, 116,  195],
        'Filler02':             [ 3645, 196, 3840],
    }

    header_data = {
        'Record Type':          [ 'DH1'],
        'School Code':          [ '002795'],
        'Branch Code':          [ '00'],
        'Official School Name': [ "PAUL SMITH'S COLLEGE OF THE ADIRONDACKS"],
        'Filler01':             [ ' '],
        'Standard Report Flag': [ 'D'],
        'Transmission Date':    [ transmission_date],
        'Degree Period':        [ degree_period],
        'Filler02':             [ ' '],
    }
    hdr_df = pd.DataFrame(data=header_data)

    return write_fw(out_fn, 'w', hdr_df, dv_hdr_def)


def write_DV_data(dv_df: pd.DataFrame, out_fn: str) -> int:
    """
    defines DegreeVerify columns
    writes fixed-width column file

    Arguments:
    dv_df: DegreeVerify dataframe
    out_fn: output file name

    Return value:
    error code from write_fw()
    """

    # column definitions
    #   'column_name': [column_width, column_start_position, column_end_position]
    dv_col_def = {
        'Record Type': [  3, 1, 3],
        'Student SSN': [  9, 4, 12],
        'First Name': [  40, 13, 52],
        'Middle Name': [  40, 53, 92],
        'Last Name': [  40, 93, 132],
        'Name Suffix': [  5, 133, 137],
        'Previous Last Name': [  40, 138, 177],
        'Previous First Name': [  40, 178, 217],
        'Date of Birth': [  8, 218, 225],
        'College Student ID': [  20, 226, 245],
        'Filler1': [  59, 246, 304],
        'Degree Level Indicator': [  1, 305, 305],
        'Degree, Certificate, or Credential Title': [  80, 306, 385],
        'School/College/Division Awarding Degree': [  50, 386, 435],
        'Joint Institution/College/School/Division Name': [  60, 436, 495],
        'Date Degree, Credential, or Certificate Awarded': [  8, 496, 503],
        'Filler02': [  80, 504, 583],
        'Major Course of Study 1': [  80, 584, 663],
        'Major Course of Study 2': [  80, 664, 743],
        'Major Course of Study 3': [  80, 744, 823],
        'Major Course of Study 4': [  80, 824, 903],
        'Filler03': [  160, 904, 1063],
        'Minor Course of Study 1': [  80, 1064, 1143],
        'Minor Course of Study 2': [  80, 1144, 1223],
        'Minor Course of Study 3': [  80, 1224, 1303],
        'Minor Course of Study 4': [  80, 1304, 1383],
        'Filler04': [  160, 1384, 1543],
        'Major Option 1': [  80, 1544, 1623],
        'Major Option 2': [  80, 1624, 1703],
        'Filler05': [  160, 1704, 1863],
        'Major Concentration 1': [  80, 1864, 1943],
        'Major Concentration 2': [  80, 1944, 2023],
        'Major Concentration 3': [  80, 2024, 2103],
        'Filler06': [  280, 2104, 2383],
        'NCES CIP Code for Major 1': [  6, 2384, 2389],
        'NCES CIP Code for Major 2': [  6, 2390, 2395],
        'NCES CIP Code for Major 3': [  6, 2396, 2401],
        'NCES CIP Code for Major 4': [  6, 2402, 2407],
        'Filler07': [  20, 2408, 2427],
        'NCES CIP Code for Minor 1': [  6, 2428, 2433],
        'NCES CIP Code for Minor 2': [  6, 2434, 2439],
        'NCES CIP Code for Minor 3': [  6, 2440, 2445],
        'NCES CIP Code for Minor 4': [  6, 2446, 2451],
        'Filler08': [  120, 2452, 2571],
        'Academic Honors': [  50, 2572, 2621],
        'Filler09': [  196, 2622, 2817],
        'Honors Program': [  50, 2818, 2867],
        'Filler10': [  100, 2868, 2967],
        'Other Honors': [  150, 2968, 3117],
        'Attendance From Date': [  8, 3118, 3125],
        'Attendance To Date': [  8, 3126, 3133],
        'FERPA Block': [  1, 3134, 3134],
        'School Financial Block': [  1, 3135, 3135],
        'Filler11': [  100, 3136, 3235],
        'Name of Institution Granting Degree': [  50, 3236, 3285],
        'Reverse Transfer Flag': [  1, 3286, 3286],
        'Certificate Type': [  1, 3287, 3287],
        'Filler12': [  553, 3288, 3840],
        }
    
    return write_fw(out_fn, 'a', dv_df, dv_col_def)


def write_DV_trailer(out_fn: str, detail_record_count:int) -> int:
    """
    defines DegreeVerify trailer columns
    writes fixed-width column file

    Arguments:
    out_fn: output file name
    detail_record_count: number of student detail records

    Return value:
    error code from write_fw()
    """

    # column definitions
    #   'column_name': [column_width, column_start_position, column_end_position]
    dv_tlr_def = {
        'Record Type':          [    3,   1,    3],
        'Total Record Count':   [   10,   4,   13],
        'Filler01':             [ 3827,  14, 3840],
    }

    total_record_count = detail_record_count + 2

    trailer_data = {
        'Record Type':          [ 'DT1'],
        'Total Record Count':   [ str(total_record_count)],
        'Filler01':             [ ' '],
    }
    tlr_df = pd.DataFrame(data=trailer_data)

    return write_fw(out_fn, 'a', tlr_df, dv_tlr_def)


def main(start_date_str:str, end_date_str:str):
    today = dt.datetime.today()
    today_str = today.strftime("%Y%m%d_%H%M")

    # Edit these 3 lines
    start_date = np.datetime64(start_date_str)
    end_date = np.datetime64(end_date_str)
    dataset_description = f"Degree Completions {start_date_str} to {end_date_str}"

    df = create_DV_df(start_date, end_date)
    logger.info(f"{df.shape=}")
    # logger.debug(df.head())

    base_fn = f'PaulSmithsCollege_DegreeVerify_{today_str}'
    xl_output = f'{base_fn}.xlsx'
    with pd.ExcelWriter(xl_output) as writer:  
        df.to_excel(writer, sheet_name='DV_file')
        logger.info(f"Excel file written: {xl_output}")
    
    lw = write_DV_header(f'{base_fn}.txt', today.strftime("%Y%m%d"), dataset_description)
    logger.info(f"{lw} header line written to {base_fn}.txt")
    lw = write_DV_data(df, f'{base_fn}.txt')
    logger.info(f"{lw} lines written to {base_fn}.txt")
    lw = write_DV_trailer(f'{base_fn}.txt', lw)
    logger.info(f"{lw} trailer line written to {base_fn}.txt")


if __name__ == "__main__":
    logger.info(f"Begin: {__file__}")
    logger.info(f'cwd: {os.getcwd()}')

    parser = ap.ArgumentParser(description="Create DegreeVerify file for National Student Cleqaringhouse.")
    parser.add_argument("start_date", type=str, help="date after previous degree submissions, ex: 2022-01-15 ")
    parser.add_argument("end_date", type=str, help="date after degree conferral, ex: 2022-05-15 ")

    args = parser.parse_args()
    start_date_str = args.start_date
    end_date_str = args.end_date

    logger.info(f'command line arguments: {start_date_str=}, {end_date_str=}')

    main(start_date_str, end_date_str)

    logger.info(f"End: {__file__}")

