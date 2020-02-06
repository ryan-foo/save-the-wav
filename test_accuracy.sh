python3 test_streaming_accuracy.py -- \
--wav=/tmp/streaming_test.wav \
--ground-truth=/tmp/streaming_test_labels.txt --verbose \
--model=/tmp/ConvNet.pb \
--labels=/tmp/speech_commands_train/conv_labels.txt \
--clip_duration_ms=1000 --detection_threshold=0.70 --average_window_ms=500 \
--suppression_ms=500 --time_tolerance_ms=1500