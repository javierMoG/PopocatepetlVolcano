import pandas as pd
import obspy 
import os
import re
import logging
from datetime import datetime

#os.chdir(os.path.normpath(os.getcwd() + os.sep + os.pardir))
os.chdir(os.getcwd()+'\\PopocatepetlVolcano')



def data_day_processing(Year,Month,Day):
    paths = os.listdir("data/"+Year+"/"+Month+"/"+Day)
    paths_PPPP = [x for x in paths if re.match(r'^{}'.format('CN.PPPP'), x)]
    bandera = True
    if len(paths_PPPP)==24:
        k = 1
        for i in paths_PPPP:
            st = obspy.read("data/"+Year+"/"+Month+"/"+Day+"/"+i)
            st = st.select(component='Z')
            if len(st)==1:
                if st[0].stats.npts==720000:
                    if k == 1:
                        st_final = st[0].resample(sampling_rate = 100)
                    else:
                        st_final = st_final + st[0].resample(sampling_rate = 100)
                    k = k+1
                else:
                    bandera = False
                    logging.error("El archivo "+i+" esta incompleto")
            else:
                bandera = False
                logging.error("No hay componente Z en el archivo "+i)

        
        if bandera:
            st_final.write("data/clean_data/"+Year+"/CN_PPPP_HHZ_"+Year+"_"+Month+"_"+Day+".sac", format = 'sac')   
    else:
        logging.error("El dia no esta completo en el dia "+Year+"_"+Month+"_"+Day)



            

for Year in ['2023']:
    # Setup Logging
    now = datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")
    log_prep_file_name = f"logs/{date_time}_prep.log"
    logging.basicConfig(
        filename=log_prep_file_name,
        level=logging.DEBUG,
        filemode='w',
        format='%(name)s - %(levelname)s - %(message)s')
    
    Months = os.listdir("data/"+Year) 
    for month in Months:
        Days = os.listdir("data/"+Year+"/"+month)
        for day in Days:
            data_day_processing(Year,month,day)
        