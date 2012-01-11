#proc areadef
  rectangle: 1 1 6 4
  xrange: 0 55
  yrange: 0 250
 
#proc yaxis
  label: Execution Time (sec)
  labeldetails: size=20 adjust=-0.5,0
  stubs: inc 50
  stubdetails: size=15
#proc xaxis
  label: #rows in W (#columns in D): d (million)
  labeldetails: size=20 adjust=0,-0.2
  stubdetails: size=15
  stubs: inc 10
                                                                                
#proc getdata
  delim: tab
  showresults: yes
  file: ./mmult_cmp3.dat
                                                                                
#proc lineplot
  xfield: 1
  yfield: 7
  legendlabel: CPMM
  pointsymbol: shape=square  radius=0.07
 
#proc getdata
  delim: tab
  showresults: yes
  file: ./mmult_cmp3.dat
                                                                                
#proc lineplot
  xfield: 1
  yfield: 11
  legendlabel: RMM
  pointsymbol: shape=triangle style=fill  radius=0.07

#proc legend
  location: 4.5 1.8
  seglen: 0.3
  textdetails: size=20
