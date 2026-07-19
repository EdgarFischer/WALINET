Für paper:

* Walinet 3T
* Inference ohne Masken / L2 Operator - reduced failure cases
* Simulationsstatistik gegen echte in vivo und pathologien validieren -> boxplots etc
* Zeigen dass man auf pathologien verallgemeinert!
* Jetzt mit LCModel basen!
* Statt den exact modes LCModel basis nehmen! Ist besser an die Realität angepasst
* simulations on the fly!
* Extrapolation warning masks after fitting!
* Bei inference lipid proj nicht mehr berechnen zu müssen spart Zeit!
* Tumor metaboliten nur mit gewisser Wahrscheinlichkeit überhaupt ziehen
* Parameter an ALLE Lcmodel ausgaben anpassen: FWHM, metabos, etc etc
* Man kann max lipid und max water durch L2 und HSVD approximieren um die simulationsparameter zu rechtfertigen.
* Für Extrapolation warnings kann man für max water / max lipid quotient bilden zu max des gesamten spektrum im vgl zu max des spektrums nach abzug der walinet baseline
* Für paper kann ich das auch schon für FWHM etc machen, und amplituden mit dem was aus LCModel kommt!
* FWHM als neuen parameter einführen peak width entfernen
* FWHM als abgeschnittene Normalverteilung! negative neu ziehen
* Gleiches bei frequncy shifts, außerhalb der grenzen neu ziehen 
* Verteilug (normal etc.) mit Histogrammen begründen! Zeigen das Simulationsverteilung in vivo abdeckt - 1 Mio spektren generieren und histogam drüber hauen
* ALLES Normalverteilt mit std 2 IQR und mittelwert = median
* realistische parameter verteilungen : Noramal
* FWHM jetzt als parameter der die alten ersetzt!

* Tumor Verteilung und healthy separat, im training zufällig ziehen aus welcher verteilung die  metabos gezogen werden.

Bemerkung: Tumor basis 7T ist vollständig und kann ich so nehmen.