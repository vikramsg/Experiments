import seamsh
from seamsh.geometry import CurveType
import numpy as np
from osgeo import osr
from seamsh import _tools

def add(fileName, points):
    op=open(fileName,'a')
    for i in points:
      pointNo = int(i[0])
      st = "{" + str(i[2]) + ", \t" + str(i[1]) + ", \t 0.0, 1.0};"
      op.write("Point("+str( pointNo )+")="+st+'\n')
    op.close()


def construct(fileName, points):
    op=open(fileName,'w')
    op.close()
    add(filename, points)

domain_srs = osr.SpatialReference()
domain_srs.ImportFromEPSG(4326)
domain = seamsh.geometry.Domain(domain_srs)

coastlines_path = "./down_data/ne_110m_coastline/ne_110m_coastline.shp" 

domain.add_boundary_curves_shp(coastlines_path,
                               "featurecla", CurveType.POLYLINE)


# Display geometry features here
#print(domain._interior_points)
#print(domain._curves)
#print(domain._interior_curves)

pts = []
cid = 0
for curve in _tools.chain(domain._curves, domain._interior_curves):
    n       = curve.points.shape[0]
    for j in range(n):
        ln      = np.array([cid, 0., 0.])
        ln[1:3] = curve.points[j] 

        pts.append(ln)

        cid     = cid + 1

filename = "earth.geo"
construct(filename, pts)
