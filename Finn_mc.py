new_zs = []

# THIS PART IS IN MY LARGER FUNCTION…
        
# first vary the redshifts --> get new state of the galaxies
for x in tqdm(range(len(data))):
    # check if spec threshold or if phot 
    if specDetermine(data["qf"].iloc[x]):
           # keep z the same 
           new_zs.append(data["z"].iloc[x])
    else:
          # draw from full pz 
           new_zs.append(genFromPZ(getCPZ(data["ID"].iloc[x],pz_data),zref))
        
# keep track of zs for each iteration
all_zs.append(new_zs)


# THESE ARE THE HELPER FUNCTIONS 

# decide wether or not to vary the galaxy
def specDetermine(qf):
    # phot always vary 
    if qf == -99:
        return False
    # create uniform
    test = random.uniform(0,1)
    
    # get level
    lev = getWeight(getDig(qf))
    
    # perform comparison 
    if test > lev:
        return False
    else:
        return True

# get correct cosmos PZ 
def getCPZ(ID,pz_data):
    # return correct pz 
    return pz_data[int(ID)][1:1002]
    
    
# get random z from the pz for that galaxy       
def genFromPZ(pz,ref):
    # create fit 
    fit = sc.interpolate.CubicSpline(ref,pz)
    
    # create more continuous func from fit 
    more = np.linspace(min(ref),max(ref),100000)
    weights = fit(more)/(len(more)/1001) # normalize!
    
    # return random weighted from pz
    return random.choices(more,weights=weights,k=1)[0]
        
        
# get weighting 
def getWeight(qflag):
    if qflag == 0 or qflag == 1:
        raise ValueError("BAD FLAG!")
    elif qflag == 2 or qflag == 9:
        return np.random.normal(loc=0.69, scale=0.05, size=1)[0] # based on VUDS from Brain
    elif qflag == 3 or qflag == 4:
        return np.random.normal(loc=0.993,scale=0.005,size=1)[0] # also from vuds
    else:
        print(qflag)
        raise ValueError("DIGIT NOT IN VUDS SCHEME")
    
# get single digit of qflag 
def getDig(qflag):
    # turn into string
    qflag = str(qflag)
    # #.# for qflag
    if len(qflag) == 3:
        return int(qflag[0])
    # ##.# for q flag
    elif len(qflag) == 4:
        return int(qflag[1])
    # ###.# for qflag
    else:
        return int(qflag[2])
