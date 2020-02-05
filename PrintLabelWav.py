import label_wav

(label_wav.label_wav(wav='output.wav', 
            labels='conv_labels.txt',
            graph='ConvNet.pb', 
            input_name='wav_data:0', 
            output_name='labels_softmax:0', 
            how_many_labels=1))