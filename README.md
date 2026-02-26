RC Design Copilot (ACI 318-19) — Streamlit Demo for UTC

University: The University of Tennessee at Chattanooga (UTC)
Department: Civil & Chemical Engineering
Courses: ENCE 2530 and ENCE 3680
App Type: Teaching demo (AI agent + modular “skills”) built with Streamlit


PURPOSE
------------------------------------------------------------------
This application demonstrates how AI agents and agent skills can
support reinforced concrete (RC) design tasks taught in undergraduate
civil engineering courses at UTC.

The app accepts a short natural-language prompt (for example:
“Design a 12x24 beam for Mu and Vu…”) and routes the request to
specialized skills.

Design intent:
- Follow ACI 318-19 reinforced concrete design concepts.
- Implement methods consistent with classical reinforced concrete
  design textbooks (strain compatibility, rectangular stress block,
  strength reduction factors, and standard beam/column/slab workflows).

IMPORTANT:
This is a TEACHING DEMO ONLY.
It is NOT a complete design program and must NOT be used for
final engineering design, construction documents, or stamping.


WHAT THE AGENT CAN DO (SKILLS)
------------------------------------------------------------------

1) Beam Flexural Design (Rectangular and T-Beam)
- Flexural design for rectangular beams and T-beams
- Handles positive moment (bottom tension steel)
- Handles negative moment (top tension steel)
- Searches bar size and quantity to satisfy phi*Mn >= Mu
- Reports strain-based phi (teaching style)

T-beam support:
- Manual bf mode (user provides bf - effective flange width)
- Optional “Auto beff” mode:
    be = min(Ln/8, 8hf, sw/2)
    bf = bw + 2be

  In this app:
    Ln = longitudinal span in beam direction (use clear span if available)
    sw = clear distance between adjacent beam webs
         (NOT transverse span)

2) Beam Shear Design
- Computes concrete shear strength Vc
- Designs stirrup spacing for Vs demand
- Includes a teaching minimum reinforcement trigger
- Can run as shear-only mode

3) Short Tied Column Design
- Teaching interaction (Pn-Mn) approach using strain compatibility
- Perimeter reinforcement layout
- Checks typical column steel ratio range
- Reports tie spacing limit (teaching)

4) One-way Slab Design (12-in Strip)
- Uses a 1-ft strip method
- Computes Mu using simple support assumptions
- Searches #4/#5 bar spacing to satisfy phi*Mn >= Mu


COVER DEFINITION USED IN THIS APP
------------------------------------------------------------------
Cover = clear cover to the OUTSIDE surface of stirrups or ties,
measured from the concrete surface to the outside surface of
transverse reinforcement.

Effective depth approximation used:
d ≈ h − cover − stirrup_dia − 0.5*bar_dia


INSTALLATION
------------------------------------------------------------------
Requirements:
- Python 3.10+ recommended
- Streamlit

Install:
    pip install -r requirements.txt

Run:
    streamlit run app.py


EXAMPLE PROMPTS
------------------------------------------------------------------

Beam Flexure (Rectangular)
Design a 12x24 rectangular beam, cover 1.5 in, fc=4 ksi, fy=60 ksi,
Mu=180 kip-ft, positive moment, use #8 max bars.

Beam Flexure (T-beam, manual bf)
Design a T-beam bw=12 in h=28 in bf=48 in hf=4 in, cover 1.5 in,
fc=4 ksi, fy=60 ksi, Mu=220 kip-ft, positive moment.

Beam Flexure (T-beam, Auto beff)
Design a T-beam bw=12 in h=28 in hf=4 in, auto beff,
Ln=24 ft, sw=72 in, cover 1.5 in, fc=4 ksi, fy=60 ksi,
Mu=220 kip-ft, positive moment.

Beam Shear (standalone)
Design shear for a 12x24 beam, cover 1.5 in, fc=4 ksi, fy=60 ksi,
Vu=70 kips, main bars #8, stirrups #3.

Column (tied)
Design a 16x16 tied column, cover 1.5 in, fc=5 ksi, fy=60 ksi,
Pu=350 kips, Mu=120 kip-ft, use #8 bars, ties #3.

One-way slab
One-way slab design: t=8 in, cover 0.75 in, fc=4 ksi, fy=60 ksi,
L=15 ft, wu=120 psf, simply supported.


SCOPE AND TEACHING SIMPLIFICATIONS
------------------------------------------------------------------
- Focuses on strength design concepts used in ACI 318-19.
- Not all detailing, spacing, serviceability, and constructability
  checks are implemented.
- T-beam negative moment is handled using a simplified teaching
  assumption (compression primarily in the web).
- Auto beff uses simplified common limits; full ACI provisions
  should be consulted for real design.


ACADEMIC USE (UTC)
------------------------------------------------------------------
This application is intended for classroom demonstrations and labs in:

- ENCE 2530
- ENCE 3680

Students should:
- Compare agent output with hand calculations
- Identify assumptions and limitations
- Document inputs and outputs
- Reflect on AI-assisted engineering workflows


DISCLAIMER
------------------------------------------------------------------
Educational use only.
This app is subject to change and modification anytime.
No warranty is provided.
The authors and the University of Tennessee at Chattanooga assume
no liability for any use outside instruction and learning activities.
