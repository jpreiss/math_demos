ogd.mp4: fast.mp4 slow.mp4
	ffmpeg -i fast.mp4 -i slow.mp4 -filter_complex hstack=inputs=2 $@

fast.mp4: online_gradient_descent.py
	python $< 1.1 1.0 $@

slow.mp4: online_gradient_descent.py
	python $< 1.1 0.04 $@
