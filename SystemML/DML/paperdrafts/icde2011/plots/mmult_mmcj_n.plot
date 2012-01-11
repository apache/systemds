#proc areadef
  rectangle: 1 1 6 4
  xrange: 0 21
  yrange: 0 2500
 
#proc yaxis
  label: Execution Time (sec)
  labeldetails: size=20 adjust=-0.5,0
  stubs: inc 500
  stubdetails: size=15
#proc xaxis
  label: #rows in V: d (million)
  labeldetails: size=20 adjust=0,-0.2
  stubdetails: size=15
  stubs: inc 5
                                                                                
#proc getdata
  delim: tab
  showresults: yes
  file: ./mmult_mmcj_n.dat
                                                                                
#proc lineplot
  xfield: 1
  yfield: 8
  legendlabel: no aggregator
  pointsymbol: shape=square  radius=0.07
 
#proc getdata
  delim: tab
  showresults: yes
  file: ./mmult_mmcj_n.dat
                                                                                
#proc lineplot
  xfield: 1
  yfield: 9
  legendlabel: with aggregator
  pointsymbol: shape=triangle style=fill  radius=0.07

#proc legend
  location: 4.3 1.8
  seglen: 0.3
  textdetails: size=20
