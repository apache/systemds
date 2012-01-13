#proc areadef
  rectangle: 1 1 6 4
  xrange: 0 1600
  yrange: 0 400
 
#proc yaxis
  label: Execution Time (sec)
  labeldetails: size=20 adjust=-0.5,0
  stubs: inc 50
  stubdetails: size=15
#proc xaxis
  label: #rows and #columns in G (thousand)
  labeldetails: size=20 adjust=0,-0.2
  stubdetails: size=15
  stubs: inc 400
                                                                                
#proc getdata
  delim: tab
  showresults: yes
  file: ./pagerank_local.dat
                                                                                
#proc lineplot
  xfield: 1
  yfield: 4
  legendlabel: DML PageRank
  pointsymbol: shape=square  radius=0.07
 
#proc legend
  location: 1.5 3.9
  seglen: 0.3
  textdetails: size=20
