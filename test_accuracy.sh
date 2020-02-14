python3 test_streaming_accuracy.py -- \
--wav=/streaming_test.wav \
--ground-truth=/streaming_test_labels.txt --verbose \
--model=/models/ConvNet_10Classes_070220.pb \
--labels=/models/conv_labels.txt \
--clip_duration_ms=1000 --detection_threshold=0.70 --average_window_ms=500 \
--suppression_ms=500 --time_tolerance_ms=1500