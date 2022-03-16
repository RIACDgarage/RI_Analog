inverter design example
.option ACCT NOMOD

*** models and parameters
.include tsmc018.m
* include action.txt file. It contains value of w0 and w1 of MOS design
.include action.txt
.option temp=27c
.param vdd=1.5

*** circuit
vsupply vdd 0 vdd
vpulse inpre 0 dc 0 pulse(0 vdd 1n 200p 200p 4.8n 10n)
m2 in inpre 0 0 nfet w='w0/3' l=180n
m3 in inpre vdd vdd pfet w='w1/3' l=180n
m0 outn in 0 0 nfet w=w0 l=180n
m1 outp in vdd vdd pfet w=w1 l=180n
c0 out 0 100f
vn out outn 0
vp outp out 0

*** current probe and raw data to be saved
.probe I(vn) I(vp) I(c0) I(vsupply)
.save in out i(vn) i(vp) i(c0) i(vsupply)

*** measurement and data
*.plot i(vsupply) i(c0) i(vn) i(vp)
.meas tran falltime TRIG v(out) VAL='vdd*0.9' FALL=last TD=15n 
+                   TARG v(out) VAL='vdd*0.1' FALL=last
.meas tran absfall param='abs(falltime)'
.meas tran risetime TRIG v(out) VAL='vdd*0.1' RISE=last TD=10n 
+                   TARG v(out) VAL='vdd*0.9' RISE=last
.meas tran absrise param='abs(risetime)'
.meas tran worsttime param='(absrise > absfall) ? absrise : absfall'
.meas tran tDiffPercent param='abs(absrise - absfall) / worsttime * 100'
*.meas tran IdotT INTEG i(vsupply) FROM=0ns TO=20ns
.meas tran iavg AVG i(vsupply)
*.meas tran power2 param='IdotT/20n*vdd'
.meas tran power param='abs(iavg)*vdd'
.meas tran speedPerPower param='worstTime/power*1e6'


.tran 10p 20n

.end
