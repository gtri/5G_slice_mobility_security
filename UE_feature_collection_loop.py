from websocket import create_connection
import json
import time
import itertools

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


def ue_bitrate(j_res, ran_id):
    samples = {}
    for i in range (0,len(j_res['ue_list'])):
        ue = j_res['ue_list'][i]
        imsi = ue['imsi']
        values = [ue['ul_bitrate'],ue['dl_bitrate']]
        samples[imsi] = values
    return samples


def ue_stats(j_res, ran_id):
    samples = {}
    for i in range (0,len(j_res['ue_list'])):
        ue = j_res['ue_list'][i]
        imsi = ue['imsi']
        values = []
        values.append(ue['dl_rx_count'])
        values.append(ue['dl_retx_count'])
        values.append(ue['dl_err_count'])
        values.append(ue['ul_tx_count'])
        values.append(ue['ul_retx_count'])
        samples[imsi] = values
    return samples

def ue_pdn(j_res, ran_id):
    samples = {}
    for i in range (0,len(j_res['ue_list'])):
        ue = j_res['ue_list'][i]
        imsi = ue['imsi']
        internet_count = 0
        ims_count = 0
        slice3_count = 0
        slice4_count = 0
        slice5_count = 0
        slice6_count = 0
        if 'pdn_list' in ue:
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
        samples[imsi] = slices
    return samples


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


def read_thread(enb_ip, ue_sample_stack):
    ws = None
    try:
        ws = create_connection('ws://%s:9002' % enb_ip)
        ws.recv()
        ws.send('{"message":"config_get"}')
        result =  ws.recv()
        j_res = json.loads(result)
        ran_id = get_ran_id(j_res)
        lte_cells = get_lte_cells(j_res)
        nr_cells = get_nr_cells(j_res)
        ws.send('{"message":"stats"}')
        result =  ws.recv()
        j_res = json.loads(result)

        ws.send('{"message":"ue_get" ,"stats":true}')
        result =  ws.recv()
        j_res = json.loads(result)

        timestamp = time.time()
        # imsi_list = imsi_id(enb_ip)
        # imsi_samples = {imsi:[] for imsi in imsi_list}
        bit_rate_samples = ue_bitrate(j_res, ran_id)
        stat_samples = ue_stats(j_res, ran_id)
        pdn_samples = ue_pdn(j_res, ran_id)
        list_of_samples = [pdn_samples, stat_samples, bit_rate_samples] 
        for k in set().union(*list_of_samples):
            ue_sample_stack[k][timestamp] = list(itertools.chain(*[d.get(k) for d in list_of_samples]))
        # imsi_samples = {
        #     k: {timestamp: list(itertools.chain(*[d.get(k) for d in list_of_samples]))}
        #     for k in set().union(*list_of_samples)
        # }
        # ue_sample_stack[timestamp] = imsi_samples
        return ue_sample_stack

    except Exception as e:
        print('error here')
        print(e)
        print('enb @ %s is not connected !' % enb_ip)
        if ws:
            ws.shutdown


def read(enb_ip,ue_sample_stack):
    ue_sample_stack = read_thread(enb_ip,ue_sample_stack)
    # global enb_list
    # for ip_i in enb_list:
    #     threading.Thread(target=read_thread,kwargs=dict(enb_ip=ip_i)).start()
    return ue_sample_stack


def stack_ue_data(ue_samples, frame_size): 
    imsi = list(ue_samples.keys())[0]
    start_timestamp = list(ue_samples[imsi].keys())[0]
    stacked_features = {}
    for ue in ue_samples.keys():
        stacked_features[ue] = ([list(feature) for feature in list(ue_samples[ue].values())[0:frame_size]])
    return start_timestamp, stacked_features


def data_collection_loop():
    # Global var def (sort of) for enb ip
    enb_ip = '10.10.4.188'
    # Make an initial connection to establish the list of IMSIs 
    ws = None
    try:
        ws = create_connection('ws://%s:9002' % enb_ip)
        ws.recv()
        ws.send('{"message":"config_get"}')
        result =  ws.recv()
        j_res = json.loads(result)
        ran_id = get_ran_id(j_res)
        lte_cells = get_lte_cells(j_res)
        nr_cells = get_nr_cells(j_res)
        ws.send('{"message":"stats"}')
        result =  ws.recv()
        j_res = json.loads(result)

        ws.send('{"message":"ue_get" ,"stats":true}')
        result =  ws.recv()
        j_res = json.loads(result)
        imsi_list = imsi_id(j_res)
    except Exception as e:
        print(e)
        print('enb @ %s is not connected !' % enb_ip)
        if ws:
            ws.shutdown
    ws.shutdown

    # Empty dict for running UE samples 
    ue_sample_stack = {list(imsi.values())[0]:{} for imsi in imsi_list}
    # Timing information 
    start_time = time.time()
    current_time = time.time()
    scenario_duration = 1800
    measurment_window = 5
    frame_size = 24

    # Run for the duration of the scenario (defualt of 30 min)
    while (time.time() - start_time) < scenario_duration:
        if time.time() - current_time > measurment_window:
            # Update vars 
            current_time = time.time()
            ue_sample_stack = read(enb_ip, ue_sample_stack)
            print(list(ue_sample_stack.values())[0])

        if len(list(ue_sample_stack.values())[0]) > frame_size:
            start_timestamp, stacked_features = stack_ue_data(ue_sample_stack, frame_size)
            print(stacked_features)

            for imsi in ue_sample_stack.keys():
                ue_sample_stack[imsi].pop(next(iter(ue_sample_stack[imsi])))

    return 


if __name__ == '__main__':
    data_collection_loop()