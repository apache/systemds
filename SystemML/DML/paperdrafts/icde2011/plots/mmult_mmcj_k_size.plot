#proc areadef
  rectangle: 1 1 6 4
  xrange: 100 310
  yrange: 0 25
 
#proc yaxis
  label: Size (GB)
  labeldetails: size=20 adjust=-0.5,0
  stubs: inc 5
  stubdetails: size=15
#proc xaxis
  label: #columns in V and #rows in H' (thousand)
  labeldetails: size=20 adjust=0,-0.2
  stubdetails: size=15
  stubs: inc 50
                                                                                
#proc getdata
  delim: tab
  showresults: yes
  file: ./mmult_mmcj_k_size.dat
                                                                                
#proc lineplot
  xfield: 1
  yfield: 8
  legendlabel: no aggregator
  pointsymbol: shape=square  radius=0.07
 
#proc getdata
  delim: tab
  showresults: yes
  file: ./mmult_mmcj_k_size.dat
                                                                                
#proc lineplot
  xfield: 1
  yfield: 9
  legendlabel: with aggregator
  pointsymbol: shape=triangle style=fill  radius=0.07

#proc legend
  location: 4.5 2.5
  seglen: 0.3
  textdetails: size=20
