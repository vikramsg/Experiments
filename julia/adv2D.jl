function fn(nx, ny, x, y)
  u = zeros(nx, ny)

  for i=1:nx
    for j = 1:ny
      dist     = (x[i] - 0.5).^2 + (y[j] - 0.5).^2

      u[i, j]  = exp.(-40*dist) 

      st = string(x[i]) * " " * string(y[j]) * " " * string(u[i, j]) 

      println(st)

    end
  end

  return u
end 

nx = 50; ny = 50;
startX = 0.0; stopX = 1.0;
startY = 0.0; stopY = 1.0;

x_grd  = zeros(nx)
y_grd  = zeros(ny)

edg_grd_x = range(startX, length=nx+1, stop = stopX)
edg_grd_y = range(startY, length=ny+1, stop = stopY)
for i=1:nx
  x_grd[i] = 0.5*( edg_grd_x[i] + edg_grd_x[i + 1] )
end
for j = 1:ny
  y_grd[j] = 0.5*( edg_grd_y[j] + edg_grd_y[j + 1] )
end

global u = fn(nx, ny, x_grd, y_grd)

#using Plots
#Plots.pyplot()
#for i = 1:5
#  p1 = contour(x_grd, y_grd, u)
#  plot(p1)
#  println("done "*string(i))
#end
#savefig("see.png")

