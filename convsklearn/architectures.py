a = [10,20,30]
architectures = []
for x in a:
	for i in range(3):
		architectures.append((x,)*(i+1))

print(architectures)