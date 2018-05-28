import sys
import numpy as np

def findEpisodeThreshold(cum_frames, max_frames):
	for i in xrange(len(cum_frames)):
		if cum_frames[i] > max_frames:
			return i

if __name__ == "__main__":

	if len(sys.argv) < 3:
		print "\nUsage: python obtainResults.py path_to_files num_frames"
		sys.exit(1)

	file_name  = sys.argv[1]
	num_frames = int(sys.argv[2])

	scores = []
	frames = []
	window = 100

	with open(file_name) as f:
		content = f.readlines()

	for line in content:
		scores.append(line.split(',')[1])
		frames.append(line.split(',')[2])

	scores = np.array(scores)
	frames = np.array(frames)
	scores = scores[1:-1].astype(float)
	frames = frames[1:-1].astype(float)

	cum_frames = np.cumsum(frames)

	trials_threshold = findEpisodeThreshold(cum_frames, num_frames)
	print np.average(scores[(trials_threshold - window + 1):(trials_threshold + 1)])
		#relevant_data.append(scores[idx][(trials_thresholds[idx] - window + 1):(trials_thresholds[idx] + 1)])
		#print np.around(np.average(trial), decimals=2)

