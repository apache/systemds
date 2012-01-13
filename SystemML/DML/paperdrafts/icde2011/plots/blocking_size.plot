#proc areadef
  rectangle: 1 1 6 4
  xrange: 0 8
  yrange: 0 16
 
#proc yaxis
  label: Execution Time (sec)
  labeldetails: size=20 adjust=-0.5,0
  stubs: inc 2
  stubdetails: size=15
#proc xaxis
  label: #rows in V: d (million)
  labeldetails: size=20 adjust=0,-0.2
  stubdetails: size=15
  stubs: inc 2
                                                                                
#proc getdata
  delim: tab
  showresults: yes
  file: ./blocking_size.dat
                                                                                
#proc lineplot
  xfield: 1
  yfield: 4
  legendlabel: 1000x1000
  pointsymbol: shape=triangle style=fill  radius=0.07
 
#proc getdata
  delim: tab
  showresults: yes
  file: ./blocking_size.dat

#proc lineplot
  xfield: 1
  yfield: 5
  legendlabel: 100x100
  pointsymbol: shape=square  radius=0.07
 
#proc legend
  location: 1.5 3.9
  seglen: 0.3
  textdetails: size=20
