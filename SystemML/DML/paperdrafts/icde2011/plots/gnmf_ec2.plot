#proc areadef
  rectangle: 1 1 6 4
  xrange: 0 5100
  yrange: 0 4400
 
#proc yaxis
  label: Execution Time (sec)
  labeldetails: size=20 adjust=-0.5,0
  stubs: inc 500
  stubdetails: size=15
#proc xaxis
  label: #nonzeros in V (million)
  labeldetails: size=20 adjust=0,-0.2
  stubdetails: size=15
  stubs: inc 1000
                                                                                
#proc getdata
  delim: tab
  showresults: yes
  file: ./gnmf_ec2.dat
                                                                                
#proc lineplot
  xfield: 1
  yfield: 15
  legendlabel: SystemML GNMF
  pointsymbol: shape=square  radius=0.07
 
#proc legend
  location: 1.5 3.9
  seglen: 0.3
  textdetails: size=20
