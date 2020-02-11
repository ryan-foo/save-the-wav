import pyaudio

pa = pyaudio.PyAudio()

for i in range(pa.get_device_count()):
  dev = pa.get_device_info_by_index(i)
  print((i,dev['name'],dev['maxInputChannels']))

devinfo = pa.get_device_info_by_index(0)  # Or whatever device you care about.
if pa.is_format_supported(16000,  # Sample rate
                         input_device=devinfo['index'],
                         input_channels=devinfo['maxInputChannels'],
                         input_format=pyaudio.paInt16):
  print('Your device\'s sampling rate is supported!')
