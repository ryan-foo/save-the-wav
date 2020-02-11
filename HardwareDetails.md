# Macbook Air 2015

# Raspberry Pi 3

Make sure to install portaudio19-dev.

`sudo apt-get install portaudio19-dev`

# Blue Microphone

16,000 sampling rate is supported. (Run `python3 pyaudio_channels.py`)

On Linux, run `arecord -l` to find the correct device
Note down it's card number and device number

`card 1, device 0`

```
pcm.!default {
	type asym
	capture.pcm "mic"
	playback.pcm "speaker"
}

pcm.mic {
	type plug
	slave {
		pcm "hw:<card number>,<device number>"
	}
}

pcm.speaker {
	type plug
	slave {
		pcm "hw:<card number>,<device number>"
	}
}
```
Tried:

(Change your Alsa config.)[https://www.raspberrypi.org/forums/viewtopic.php?t=136974]

`sudo nano /usr/share/alsa/alsa.conf`

You have to install jackaudio and begin running Jack.

`jackd -d alsa -r16000 -p16000 -i1`

Sound cards keep switching indices
	- unix stack exchange

Sound card - default 

## Sound modules
`sudo lsmod | grep snd`

`cd /sys/module`

