import seamsh
from seamsh.geometry import CurveType
import numpy as np
import pandas as pd
from osgeo import osr
from seamsh import _tools

def addPts(fileName, bnd):
  op=open(fileName,'a')
  op.write("// Points \n")
  for index, row in bnd.iterrows():
    pointNo = int(row["pid"])
    st = "{" + str(row["lon"]) + ", \t" + str(row["lat"]) + ", \t 0.0, 1.0};"
    op.write("Point("+str( pointNo )+")="+st+'\n')
  op.close()

def addLns(fileName, bnd):
  op=open(fileName,'a')
  op.write("\n// Lines\n")

  cid = int(bnd.iloc[0]["cid"])

  lst = []
  # Brute force approach. Not nice but will get it done
  for index, row in bnd.iterrows():
    ncid = int(row["cid"])
    pid = int(row["pid"])
    if (cid == ncid):
      lst.append(pid)
    else:
      st = "{" 
      ## Split into all terms and last term
      ## to avoid placing comma at the end
      for j in lst[:-1]:
          st = st + str(j) + ", "
      st = st + str(lst[-1]) +"};"
      op.write("Spline("+str( cid )+")="+st+'\n')
      cid = ncid
      lst = [pid]

  ## We miss out on the last list with the above logic
  ## So doing it here
  if (len(lst) > 0):
      st = "{" 
      ## Split into all terms and last term
      ## to avoid placing comma at the end
      for j in lst[:-1]:
          st = st + str(j) + ", "
      st = st + str(lst[-1]) +"};"

      op.write("Spline("+str( cid )+")="+st+'\n')

  op.close()

def construct(fileName, bnd):
  op=open(fileName,'w')
  op.close()
  addPts(filename, bnd)
  addLns(filename, bnd)

def getBoundaryDataFrame(cfile):
  domain_srs = osr.SpatialReference()
  domain_srs.ImportFromEPSG(4326)
  domain = seamsh.geometry.Domain(domain_srs)
  
  domain.add_boundary_curves_shp(cfile,
                                 "featurecla", CurveType.POLYLINE)
  
  # Display geometry features here
  #print(domain._interior_points)
  #print(domain._curves)
  #print(domain._interior_curves)
  
  pts = []
  pid = 0
  cid  = 0
  for curve in _tools.chain(domain._curves, domain._interior_curves):
      n    = curve.points.shape[0]
      for j in range(n):
          ln      = np.array([pid, cid, 0., 0.])
          ln[2:4] = curve.points[j] 
  
          pts.append(ln)
  
          pid     = pid + 1
      cid     = cid + 1
  
  bnd = pd.DataFrame(data=pts, columns=["pid", "cid", "lat", "lon"])
  bnd.pid = bnd.pid.astype(int)
  bnd.cid = bnd.cid.astype(int)

  return bnd

#filename = "earth.geo"
#construct(filename, pts)


"""
We give lonlat extent and the code should automaticlly
select only points within that extent. 
Maybe we should convert it into a dataframe and retain
curve id as well in the dataframe
Then we can slice dataframe to retain only those points
that are within the lonlat, and then use this
to both create points and Bsplines
"""
if __name__=="__main__":
    """
    """
  
    coastlines_file = "./down_data/ne_110m_coastline/ne_110m_coastline.shp" 

    cln = getBoundaryDataFrame(coastlines_file)
    print(cln.head())

    bnd = cln.loc[ ( cln["lat"] > -30 ) & ( cln["lat"] < 70 ) & 
            ( cln["lon"] > -100 ) & ( cln["lon"] < 20 )]
    print(bnd.head())

    filename = "earth.geo"
    construct(filename, bnd)

