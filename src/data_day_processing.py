import pandas as pd
import obspy 
import os
import re
import logging
import obspy.signal.interpolation as itp
import obspy.core.stream 
import obspy.core.trace
from datetime import datetime, timedelta
import numpy as np

def create_empty_day(Year, Month, Day):
    """
    Creates a seismic trace of zeros for a full day.

    Parameters:
    ----------
    Year : int or str
        The year of the date for the trace.
    Month : int or str
        The month of the date for the trace.
    Day : int or str
        The day of the month for the trace.

    Returns:
    -------
    obspy.Trace
        A `obspy.Trace` object representing an empty seismic trace for the specified day.
        The trace spans from 00:00:00 to 23:59:59.59 UTC, with a sampling rate of 100 Hz.
    """
    # Create an array of zeros with dtype=float64
    array = np.zeros(360000, dtype=np.float64)

    # Create the start trace
    st_start = obspy.core.trace.Trace(data=array)
    st_start.stats.starttime = f"{Year}-{Month}-{Day}T00:00:00Z"
    st_start.id = "CN.PPPP..HHZ"
    st_start.stats.delta = 0.01
    st_start.sampling_rate = 100

    # Create the end trace
    st_end = obspy.core.trace.Trace(data=array)
    st_end.stats.starttime = f"{Year}-{Month}-{Day}T23:00:00Z"
    st_end.id = "CN.PPPP..HHZ"
    st_end.stats.delta = 0.01
    st_end.sampling_rate = 100

    # Combine the start and end traces
    st_final = obspy.Trace.__add__(self=st_start, trace=st_end, fill_value=0.0)

    return st_final

def interpolate_hour(trace):
    """
    Combines a list of seismic trace objects into a single continuous trace by interpolating
    missing data between adjacent traces.

    Parameters:
    ----------
    trace : list of obspy.Trace
        A list containing `obspy.Trace` objects representing different parts of the seismic data. 

    Returns:
    -------
    obspy.Trace
        A single `obspy.Trace` object that results from the combination of the input traces,
        with gaps filled using interpolation.

    Notes:
    ------
    - The function initializes with the first trace in `trace` and iteratively adds each
      subsequent trace to it, using interpolation to fill gaps.
    """

    n = len(trace)
    for i in range(n):
        trace[i].data = trace[i].data.astype(np.float64)
        if i == 0:
            st_final = trace[i]
        else:
            st_final = obspy.Trace.__add__(self = st_final, trace = trace[i], fill_value ='interpolate')
    return st_final

def complete_day(Year, Month, Day, source_path, paths_PPPP):
    """
    Completes a seismic trace for a full day by reading and combining trace data 
    from multiple files and filling missing data as needed.

    Parameters:
    ----------
    Year : str
        The year of the date for the trace (e.g., "2024").
    Month : str
        The month of the date for the trace (e.g., "09" for September).
    Day : str
        The day of the month for the trace (e.g., "05").
    source_path : str
        The path where the sismograms data files are located.
    paths_PPPP : list of str
        A list of file paths containing seismic trace data for the given day.

    Returns:
    -------
    obspy.Trace
        A `obspy.Trace` object representing the combined seismic trace for the specified day.
        The trace spans from 00:00:00 to 23:59:59.99 UTC.

    Notes:
    ------
    - The function reads trace data from the provided list of file paths.
    - It selects only the 'Z' component of the seismic data and resamples the data to a sampling rate of 100 Hz.
    - If the trace for the 'Z' component is present, it combines them into a single trace.
    - If no trace is found for a particular hour, the function fills the missing hour with zeros.
    - If trace for a particular hour is incomplete, the function interpolates the missing data for that hour.
    - It ensures the trace starts at 00:00:00 UTC and ends at 23:59:59.99 UTC, adding empty data at the start or end if necessary.
    """

    k = 1
    for i in paths_PPPP:
        st = obspy.read(source_path + "/" + Year + "/" + Month + "/" + Day + "/" + i)
        st = st.select(component='Z')
        st = st.decimate(factor=2, no_filter=True)
        # Validate that we have only one measurement of component Z
        if len(st) == 1:
            st[0].data = st[0].data.astype(np.float64)  # Ensure data is float64
            if k == 1:
                st_final = st[0]
            else:
                st_final2 = st[0]
                st_final = obspy.Trace.__add__(self=st_final, trace=st_final2, fill_value=0.0)
            k = k + 1
        else:
            # Validate that we have measurements of component Z
            if len(st) > 1:
                st = interpolate_hour(st)
                if k == 1:
                    st_final = st
                else:
                    st_final2 = st
                    st_final2.data = st_final2.data.astype(np.float64)  # Ensure data is float64
                    st_final = obspy.Trace.__add__(self=st_final, trace=st_final2, fill_value=0.0)
                k = k + 1

    # Ensure the trace starts at 00:00:00 UTC
    if st_final.stats.starttime.hour != 0:
        array = np.zeros(360000, dtype=np.float64)  # Ensure data is float64
        st_start = obspy.core.trace.Trace(data=array)
        st_start.stats.starttime = f"{Year}-{Month}-{Day}T00:00:00Z"
        st_start.id = "CN.PPPP..HHZ"
        st_start.stats.delta = 0.01
        st_start.sampling_rate = 100

        st_final = obspy.Trace.__add__(self=st_start, trace=st_final, fill_value=0.0)

    # Ensure the trace ends at 23:59:59.99 UTC
    if st_final.stats.endtime.hour != 23:
        array = np.zeros(360000, dtype=np.float64)  # Ensure data is float64
        st_end = obspy.core.trace.Trace(data=array)
        st_end.stats.starttime = f"{Year}-{Month}-{Day}T23:00:00Z"
        st_end.id = "CN.PPPP..HHZ"
        st_end.stats.delta = 0.01
        st_end.sampling_rate = 100

        st_final = obspy.Trace.__add__(self=st_final, trace=st_end, fill_value=0.0)

    return st_final


def data_day_processing(Year, Month, Day, source_path, destination_path):
    """
    Processes seismic data for a given day, validating and combining trace data from
    the PPPP station, and filling in any missing data.

    Parameters:
    ----------
    Year : str
        The year of the date for the data (e.g., "2024").
    Month : str
        The month of the date for the data (e.g., "09" for September).
    Day : str
        The day of the month for the data (e.g., "05").
    source_path : str
        The path where the sismograms data files are located.
    destination_path : str
        The path where the processed data for the day will be saved.

    Returns:
    -------
    None
        The function processes and combines the seismic trace data for the given day and
        writes the final trace to a file in MiniSEED format at "destination_path".

    Functionality:
    --------------
    - It reads files from the "./{Year}/{Month}/{Day}" directory corresponding to the
      station "CN.PPPP" (a seismic station).
    - If 24 hourly measurements exist for the day, it validates the trace data for each
      hour, ensuring each measurement is complete (containing 360,000 points).
    - If the data is valid, it combines traces for the entire day.
    - If some hourly files are missing or incomplete, the function calls `complete_day`
      to fill gaps.
    - If no PPPP station data is available, the function generates an empty trace using
      `create_empty_day`.
    - The final seismic trace for the day is written to a file in the path:
      "{destination_path}/{Year}/CN_PPPP_HHZ_{Year}_{Month}_{Day}.seed".

    Notes:
    ------
    - The function ensures the trace is for the Z component, resamples it to a sampling
      rate of 100 Hz, and logs errors if data is missing or incomplete.
    - It assumes that each hourly file should contain 360,000 data points.
    - Missing hours or incomplete data trigger error logging and fallback to gap-filling.
    """

    paths = os.listdir(source_path+"/"+Year+"/"+Month+"/"+Day)
    paths_PPPP = [x for x in paths if re.match(r'^{}'.format('CN.PPPP'), x)]
    bandera = True

    # Validate that the destination path for the year exists
    os.makedirs(destination_path+"/"+Year, exist_ok=True)

    # Create an empty Trace object
    st_final = obspy.core.trace.Trace()

    # Validate if there exists a measure from the PPPP station for the day
    if len(paths_PPPP)>0: 
        # Validate that the day contains 24 hours
        if len(paths_PPPP)==24:
            i = 0
            while i < 24 and bandera:
                st = obspy.read(source_path+"/"+Year+"/"+Month+"/"+Day+"/"+paths_PPPP[i])
                st = st.select(component='Z')
                st = st.decimate(factor=2, no_filter=True)
                # Validate that we have only one measurement of component Z
                if len(st)==1:
                    # Validate that the hour measurement is complete
                    if st[0].stats.npts==360000:
                        if i == 0:
                            st_final = st[0]
                        else:
                            st_final = st_final + st[0]
                        i = i+1
                    else:
                        bandera = False
                        logging.error("El archivo "+paths_PPPP[i]+" esta incompleto")
                else:
                    bandera = False
                    if len(st)>1:
                        logging.error("Hay múltiples mediciones de la componente Z en el archivo "+paths_PPPP[i])
                    else:
                        logging.error("No hay componente Z en el archivo "+paths_PPPP[i])

            if not(bandera):
                st_final = complete_day(Year, Month, Day, source_path, paths_PPPP)

        else:
            logging.error("El dia "+Year+"_"+Month+"_"+Day+" no está completo, tiene "+str(len(paths_PPPP))+" horas en lugar de 24")
            st_final = complete_day(Year, Month, Day, source_path, paths_PPPP)
    else:
        logging.error("El dia "+Year+"_"+Month+"_"+Day+ " no tiene archivos para la estación PPPP")
        st_final = create_empty_day(Year, Month, Day)

    st_final.write(destination_path + "/" + Year + "/CN_PPPP_HHZ_" + Year + "_" + Month + "_" + Day + ".seed", format="MSEED") 

def process_date_range(start_date, end_date, logs_path, source_path, destination_path):
    """
    Iterate over all calendar days from start_date to end_date (inclusive) and call
    data_day_processing for each day.

    Parameters:
    - start_date: str | datetime.date | datetime.datetime
        Start date as "YYYY-MM-DD" string or a date/datetime object.
    - end_date: str | datetime.date | datetime.datetime
        End date as "YYYY-MM-DD" string or a date/datetime object.
    - logs_path: str
        Path where log files will be saved.
    - source_path: str
        Source path passed to data_day_processing where raw files are located.
    - destination_path: str
        Destination path passed to data_day_processing where processed files are saved.

    Notes:
    - The function validates the range and logs exceptions without stopping the loop.
    """

    # Setup Logging
    now = datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")
    # Validate that the destination path for the year exists
    os.makedirs(logs_path, exist_ok=True)
    log_prep_file_name = f"{logs_path}/{date_time}_prep.log" #File path to almacenate the logs
    logging.basicConfig(
        filename=log_prep_file_name,
        level=logging.DEBUG,
        filemode='w',
        format='%(name)s - %(levelname)s - %(message)s')
    
    # Normalize inputs to date objects
    if isinstance(start_date, str):
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
    elif isinstance(start_date, datetime):
        start = start_date.date()
    else:
        start = start_date

    if isinstance(end_date, str):
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
    elif isinstance(end_date, datetime):
        end = end_date.date()
    else:
        end = end_date

    if start > end:
        raise ValueError("start_date must be <= end_date")

    current = start
    while current <= end:
        Year = f"{current.year:04d}"
        Month = f"{current.month:02d}"
        Day = f"{current.day:02d}"
        try:
            os.makedirs(destination_path, exist_ok=True)
            data_day_processing(Year, Month, Day, source_path, destination_path)
        except Exception as e:
            logging.exception(f"Error processing {Year}-{Month}-{Day}: {e}")
        current += timedelta(days=1)

# process_date_range("2023-12-29", "2023-12-30", "D:/Popocatepetl/data", "D:/Popocatepetl/processed_data")
