# Appunti — Entangletronica (debug fisico)

## IL PROBLEMA TROVATO (la cosa importante)

L'elettrone non si muoveva. Centro di massa fermo a x=25 dopo 300 step.

**Causa**: griglia con Δx = 8 nm troppo grossolana per il pacchetto.

Il pacchetto gaussiano ha momento k0 = 0.70 nm⁻¹. Il criterio di Nyquist
richiede Δx ≤ π/k0 = 4.49 nm. Con Δx = 8 nm la modulazione e^{ikx} viene
piegata dall'FFT: il picco in k-spazio appariva a kx = -0.086 (quasi zero),
quindi il pacchetto "vedeva" un momento nullo e restava fermo.

**Fix**: Δx = 4 nm. Con questa griglia il picco in k-spazio è a 0.699 = k0 ✓
Velocità misurata ~0.2 nm/step, coerente con v = k0·dt = 0.7·0.30 = 0.21.

## SISTEMA DI UNITÀ (naturali: ħ = m = 1)

- Lunghezze in nm
- Energia in meV → convertita in unità naturali:
  MEV_TO_NAT ≈ 5.5e-4
- Tempo: TIME_UNIT ≈ 3.6e-16 s per unità di tempo simulata

Conversione energia: E_nat = E_meV × 5.5e-4

Verifica: k0 = 0.7 → E_kin = 0.5·k0² = 0.245 nat = 445 meV ✓
(energia cinetica ~0.5 eV, muri da 3 meV sono una perturbazione debole: giusto)

## PARAMETRI FISICI (InGaAs 2DEG)

- m* = 0.042 m₀
- x0 = 25 nm (partenza), s = 12 nm (spread), k0 = 0.7 nm⁻¹
- v = ħk0/m* ≈ 2.7e5 m/s (ballistico)
- Transito interferometro (~300 nm) ≈ 1 ps → scala di commutazione

## VINCOLI NUMERICI (da rispettare sempre)

- Δx ≤ π/k0 → con k0 = 0.7: Δx ≤ 4.5 nm
- λ_deBroglie = 2π/k0 = 9 nm ≥ 2·Δx = 8 nm ✓
- Se alzi k0 a 1.4 (per andare più veloce): λ = 4.5 nm, serve Δx = 2 nm
  (attenzione: costo quadruplo in memoria/FFT)

## STATO ATTUALE (ultimo test)

Griglia 240×120, Δx = 4 nm, dt = 0.30, Nt = 200
- Norma = 1.000000 (unitario ✓)
- Centroidi x: n=60→37, n=120→49, n=180→61 → l'elettrone vola ✓
- MA: troppo lento per il MZ completo (200 step = 40 nm percorsi)
- I rivelatori (a destra del dominio) non vedono ancora niente

## PROSSIMO PASSO

1. Ridurre il dominio a ~256 nm (ora 480 nm in x) → meno step per transito
2. Alzare k0 a 1.4 con Δx = 2 nm (o accettare dominio corto con k0 = 0.7)
3. Verificare lo splitter: la probabilità deve dividersi 50/50 ai due bracci
4. Poi sweep di fase → caratteristica di trasferimento oscillante
5. Poi gate dinamico

## REGOLA D'ORO

Prima di pubblicare una figura: controlla SEMPRE che
- la norma sia 1.000000000000
- il centro di massa si muova nella direzione giusta alla velocità giusta
- il picco in k-spazio del pacchetto coincida con k0

Se l'elettrone non vola, nessun risultato di interferenza è attendibile.
