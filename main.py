from utils import split_data

snr_image = ['0dB_tf', '2dB_tf', '-2dB_tf', '4dB_tf', '-4dB_tf',
             '6dB_tf', '-6dB_tf', '8dB_tf', '-8dB_tf', '10dB_tf', '-10dB_tf']
snr_signal = ['0dB_time', '2dB_time', '-2dB_time', '4dB_time', '-4dB_time',
              '6dB_time', '-6dB_time', '8dB_time', '-8dB_time', '10dB_time', '-10dB_time']
for s in snr_signal:
    split_data(root="../data/time", target_root="../data/signal", snr=s, ratio=0.2)
