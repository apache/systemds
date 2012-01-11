#proc areadef
  rectangle: 1 1 6 4
  xrange: 0 2100
  yrange: 0 4100
 
#proc yaxis
  label: Execution Time (sec)
  labeldetails: size=20 adjust=-0.5,0
  stubs: inc 500
  stubdetails: size=15
#proc xaxis
  label: #nonzeros in V (million)
  labeldetails: size=20 adjust=0,-0.2
  stubdetails: size=15
  stubs: inc 500
                                                                                
#proc getdata
  delim: tab
  showresults: yes
  file: ./gnmf_cmp_all.dat
                                                                                
#proc lineplot
  xfield: 1
  yfield: 16
  legendlabel: hand-coded
  pointsymbol: shape=triangle style=fill  radius=0.07
 
#proc lineplot
  xfield: 1
  yfield: 17
  legendlabel: SystemML
  pointsymbol: shape=square  radius=0.07

#proc lineplot
  xfield: 1
  yfield: 18
  legendlabel: single node R
  pointsymbol: style=spokes shape=circle radius=0.07

#proc legend
  location: 4 3.9
  seglen: 0.3
  textdetails: size=20
