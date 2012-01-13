#proc areadef
  rectangle: 1 1 6 4
  xrange: 0 8
  yrange: 0 4000
 
#proc yaxis
  label: Execution Time (sec)
  labeldetails: size=20 adjust=-0.5,0
  stubs: inc 500
  stubdetails: size=15
#proc xaxis
  label: # documents: d (million)
  labeldetails: size=20 adjust=0,-0.2
  stubdetails: size=15
  stubs: inc 1
                                                                                
#proc getdata
  delim: tab
  showresults: yes
  file: ./dml_vs_hadoop.dat
                                                                                
#proc lineplot
  xfield: 1
  yfield: 17
  legendlabel: handcode
  pointsymbol: shape=square  radius=0.07
 
#proc getdata
  delim: tab
  showresults: yes
  file: ./dml_vs_hadoop.dat
                                                                                
#proc lineplot
  xfield: 1
  yfield: 18
  legendlabel: dml
  pointsymbol: shape=triangle style=fill  radius=0.07

#proc legend
  location: 4.3 2.5
  seglen: 0.3
  textdetails: size=20
