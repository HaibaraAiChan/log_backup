# Output	    status Field	Description
#---------------------------------------------------------------------
# currRealMem	VmRSS	    Resident set size
# peakRealMem	VmHWM	    Peak resident set size ("high water mark")
# currVirtMem	VmSize	    Virtual memory size
# peakVirtMem	VmPeak	    Peak virtual memory size

#Real memory (resident set size) is the amount of physical RAM your process is using,
# and virtual memory is the size of the memory address space your process is using.
# Linux chooses what in your virtual memory gets to reside in RAM.
# Note that in addition to your program data, these memories include the space taken up by your code itself,
# and any libraries your code is using (which may be shared by other running processes, skewing your usage)

_FIELDS = ['VmRSS', 'VmHWM', 'VmSize', 'VmPeak']


def get_memory(str1):
	'''
	returns the current and peak, real and virtual memories
	used by the calling linux python process, in Bytes
	'''

	# read in process info
	with open('/proc/self/status', 'r') as file:
		lines = file.read().split('\n')

	# container of memory values (_FIELDS)
	values = {}

	# check all process info fields
	for line in lines:
		if ':' in line:
			name, val = line.split(':')

			# collect relevant memory fields
			if name in _FIELDS:
				values[name] = int(val.strip().split(' ')[0])  # strip off "kB"
				values[name] /= 1024  # convert to MB

	# check we collected all info
	assert len(values)==len(_FIELDS)
	return print(str1+"  \n"+str(values)+"  \n")



