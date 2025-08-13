import collectd  
import json
from websocket import create_connection

def get_ran_id(j_res):
    if j_res and j_res.get('global_enb_id'):
        return str(j_res['global_enb_id']['enb_id'])
    elif j_res and j_res.get('global_gnb_id'):
        return str(j_res['global_gnb_id']['gnb_id'])
    else:
        return '-1'

def get_lte_cells(j_res):
    cells = []
    if j_res and j_res.get('cells'):
        for key in j_res.get('cells'):
            cells.append(key)
    return cells

def get_nr_cells(j_res):
    cells = []
    if j_res and j_res.get('nr_cells'):
        for key in j_res.get('nr_cells'):
            cells.append(key)
    return cells                 

def ue_count(j_res,ran_id):
    vl = collectd.Values(type='count')
    vl.plugin = 'ran_ue'
    vl.plugin_instance = 'total'
    vl.host = ran_id
    vl.type_instance = 'none'
    vl.interval = 5
    vl.dispatch(values=[len(j_res['ue_list'])])        
 

def imsi_id(j_res):
    try:
        imsi_list = []       
        ue_list=j_res['ue_list']
        for ue in ue_list:
            imsi=str(ue['imsi'])
            imsi_list.append({'imsi':imsi})
        return imsi_list
    except Exception as ex:
        print(ex)   


def ue_bitrate(j_res, imsi_list, ran_id):
    for i in range (0,len(j_res['ue_list'])):
        ue = j_res['ue_list'][i]
        imsi = ue['imsi']
        vl = collectd.Values(type='throughput')
        vl.plugin='ue'
        vl.plugin_instance=imsi
        vl.host=ran_id
        vl.type_instance='lte'
        vl.interval=5
        vl.dispatch(values=[ue['ul_bitrate'],ue['dl_bitrate']])

def ue_stats(j_res, imsi_list, ran_id):
    for i in range (0,len(j_res['ue_list'])):
        ue = j_res['ue_list'][i]
        imsi = ue['imsi']
        vl = collectd.Values(type='stats')
        vl.plugin='ue'
        vl.plugin_instance=imsi
        vl.host=ran_id
        vl.type_instance='nr'
        vl.interval=5
        values = []
        values.append(ue['dl_rx_count'])
        values.append(ue['dl_retx_count'])
        values.append(ue['dl_err_count'])
        values.append(ue['ul_tx_count'])
        values.append(ue['ul_retx_count'])
        vl.dispatch(values=values)


def ue_pdn(j_res, ran_id):
    for i in range (0,len(j_res['ue_list'])):
        ue = j_res['ue_list'][i]
        print('IMSI', ue['imsi'])
        if 'pdn_list' in ue:
            internet_count = 0
            ims_count = 0
            slice3_count = 0
            slice4_count = 0
            slice5_count = 0
            slice6_count = 0
            for pdn_session in ue['pdn_list']:
                if pdn_session['apn'] == 'internet':
                    internet_count += 1
                elif pdn_session['apn'] == 'ims':
                    ims_count += 1
                elif pdn_session['apn'] == 'slice3':
                    slice3_count += 1
                elif pdn_session['apn'] == 'slice4':
                    slice4_count += 1 
                elif pdn_session['apn'] == 'slice5':
                    slice5_count += 1
                elif pdn_session['apn'] == 'slice6':
                    slice6_count += 1
                else: 
                    pass
            slices = []
            slices.append(internet_count)
            slices.append(ims_count)
            slices.append(slice3_count)
            slices.append(slice4_count)
            slices.append(slice5_count)
            slices.append(slice6_count)
            imsi = ue['imsi']
            print('slices', slices)
            vl = collectd.Values(type='slices')
            vl.plugin='ue'
            vl.plugin_instance=imsi
            vl.host=ran_id
            vl.type_instance='nr'
            vl.interval=5
            vl.dispatch(values=slices)
