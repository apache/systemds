#proc areadef
  rectangle: 1 1 6 4
  xrange: 20 110
  yrange: 0 2000
 
#proc yaxis
  label: Execution Time (sec)
  labeldetails: size=20 adjust=-0.5,0
  stubs: inc 200
  stubdetails: size=15
#proc xaxis
  label: # workers
  labeldetails: size=20 adjust=0,-0.2
  stubdetails: size=15
  stubs: inc 20
                                                                                
#proc getdata
  delim: tab
  showresults: yes
  file: ./ec2_scale.dat
                                                                                
#proc lineplot
  xfield: 1
  yfield: 14
  legendlabel: SystemML GNMF
  pointsymbol: shape=square  radius=0.07
 
#proc legend
  location: 1.5 3.9
  seglen: 0.3
  textdetails: size=20
