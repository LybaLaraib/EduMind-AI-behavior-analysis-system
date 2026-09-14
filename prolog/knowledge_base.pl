/*
 * EduMind — Classroom Behavior Expert System
 * Full observation rules: posture, eyes, mobile, face presence
 */

:- module(knowledge_base, [
    load_expert_rules/0,
    clear_observations/0,
    assert_observation/1,
    infer_all/1
]).

:- dynamic observation/1.

load_expert_rules :- true.

clear_observations :- retractall(observation(_)).

assert_observation(Fact) :-
    atom(Fact),
    (   observation(Fact) -> true
    ;   assertz(observation(Fact))
    ).

/* Behavior rules */
sleepy_student :-
    observation(eyes_closed),
    observation(head_down).

sleepy_student :-
    observation(eyes_closed),
    observation(low_blink_rate).

distracted_student :-
    observation(looking_away),
    observation(face_detected).

distracted_student :-
    observation(looking_away),
    observation(phone_suspected).

distracted_student :-
    observation(looking_away),
    observation(head_down),
    \+ observation(eyes_closed).

posture_distracted :-
    observation(looking_away).

posture_distracted :-
    observation(head_down),
    \+ observation(eye_contact).

phone_usage_suspected :-
    observation(phone_suspected).

phone_usage_suspected :-
    observation(hand_near_face),
    observation(looking_down).

attentive_student :-
    observation(face_detected),
    observation(eye_contact),
    observation(upright_posture),
    \+ observation(looking_away),
    \+ observation(eyes_closed),
    \+ observation(phone_suspected).

partially_attentive :-
    observation(face_detected),
    \+ attentive_student,
    \+ sleepy_student,
    \+ distracted_student,
    \+ phone_usage_suspected.

disengaged_student :-
    observation(no_face),
    \+ observation(face_detected).

/* Eyes closed 2+ seconds (may occur without head_down) */
drowsy_eyes :-
    observation(eyes_closed),
    \+ observation(head_down),
    \+ phone_usage_suspected.

/* Conclusions */
attention_level(high) :- attentive_student.
attention_level(medium) :- partially_attentive.
attention_level(medium) :- distracted_student, \+ sleepy_student, \+ phone_usage_suspected.
attention_level(low) :- sleepy_student.
attention_level(low) :- drowsy_eyes.
attention_level(low) :- phone_usage_suspected.
attention_level(low) :- disengaged_student.
attention_level(low) :- distracted_student, sleepy_student.

behavior_status(attentive) :- attentive_student.
behavior_status(sleepy) :- sleepy_student, \+ phone_usage_suspected.
behavior_status(sleepy) :- drowsy_eyes.
behavior_status(distracted) :- distracted_student, \+ sleepy_student.
behavior_status(phone_use) :- phone_usage_suspected.
behavior_status(disengaged) :- disengaged_student.
behavior_status(neutral) :-
    partially_attentive,
    \+ behavior_status(attentive),
    \+ behavior_status(sleepy),
    \+ behavior_status(distracted),
    \+ behavior_status(phone_use),
    \+ behavior_status(disengaged).

risk_level(high) :- sleepy_student, phone_usage_suspected.
risk_level(high) :- disengaged_student.
risk_level(medium) :- distracted_student.
risk_level(medium) :- sleepy_student.
risk_level(medium) :- phone_usage_suspected.
risk_level(low) :- attentive_student.
risk_level(low) :-
    partially_attentive,
    \+ risk_level(medium),
    \+ risk_level(high).

recommendation('Maintain your current focus — excellent engagement with the lecture.') :- attentive_student.
recommendation('You appear tired. Consider a short break or adjust your seating for alertness.') :-
    sleepy_student, \+ phone_usage_suspected.
recommendation('Your eyes have been closed — reopen them and refocus on the screen.') :-
    drowsy_eyes, \+ recommendation(_).
recommendation('Refocus on the screen. Minimize side glances and keep notes within your primary view.') :-
    distracted_student, \+ sleepy_student.
recommendation('Put your phone away and return attention to the learning material.') :- phone_usage_suspected.
recommendation('Position yourself in front of the camera so the system can support your learning session.') :-
    disengaged_student.
recommendation('Stay engaged — re-center your gaze on the instructor or slides.') :-
    partially_attentive,
    \+ recommendation(_).

possible_reason('Stable posture, open eyes, and eye contact with the display.') :- attentive_student.
possible_reason('Eyes closed with head lowered — indicators of drowsiness.') :- sleepy_student.
possible_reason('Eyes closed for an extended period — attention is reduced.') :- drowsy_eyes.
possible_reason('Frequent gaze away from the learning screen.') :- distracted_student, \+ sleepy_student.
possible_reason('Hand activity near face suggests possible mobile device use.') :- phone_usage_suspected.
possible_reason('No face visible — student may have left the seat or camera is blocked.') :- disengaged_student.
possible_reason('Mixed signals — monitor for sustained patterns.') :-
    partially_attentive,
    \+ possible_reason(_).

infer_all(Results) :-
    resolve_attention(Attention),
    resolve_behavior(Behavior),
    resolve_risk(Risk),
    resolve_recommendation(Rec),
    resolve_reason(Reason),
    build_inference_steps(Steps),
    flag_sleepy(Sleepy),
    flag_distracted(Distracted),
    flag_phone(Phone),
    Results = _{
        attention: Attention,
        behavior: Behavior,
        risk: Risk,
        recommendation: Rec,
        reason: Reason,
        inference_steps: Steps,
        flags: _{sleepy: Sleepy, distracted: Distracted, phone: Phone}
    }.

resolve_attention(A) :- attention_level(A), !.
resolve_attention(medium).

resolve_behavior(B) :- behavior_status(B), !.
resolve_behavior(neutral).

resolve_risk(R) :- risk_level(R), !.
resolve_risk(low).

resolve_recommendation(M) :- recommendation(M), !.
resolve_recommendation('Continue monitoring your engagement.').

resolve_reason(P) :- possible_reason(P), !.
resolve_reason('Analysis complete.').

flag_sleepy(true) :- sleepy_student, !.
flag_sleepy(false).

flag_distracted(true) :- distracted_student, !.
flag_distracted(false).

flag_phone(true) :- phone_usage_suspected, !.
flag_phone(false).

build_inference_steps(Steps) :-
    findall(_{technique: T, rule: R, result: Res}, inference_step(T, R, Res), Steps).

inference_step('Unification', 'assert_observation/1', 'Runtime facts bound to observation/1 predicates.').

inference_step('Backward chaining', 'attentive_student', 'Behavior classified as attentive.') :- attentive_student.
inference_step('Modus ponens', 'attention_level(high) :- attentive_student', 'Attention level set to high.') :- attentive_student.
inference_step('Backward chaining', 'sleepy_student', 'Drowsiness inferred from eyes and posture.') :- sleepy_student.
inference_step('Backward chaining', 'drowsy_eyes', 'Eyes closed 2+ seconds — attention lowered.') :- drowsy_eyes.
inference_step('Backward chaining', 'distracted_student', 'Distraction inferred from gaze/posture.') :- distracted_student, \+ sleepy_student.
inference_step('Backward chaining', 'phone_usage_suspected', 'Possible mobile use detected.') :- phone_usage_suspected.
inference_step('Backward chaining', 'disengaged_student', 'No face visible.') :- disengaged_student.
inference_step('Unification', 'partially_attentive', 'Mixed signals — monitoring continues.') :-
    partially_attentive,
    \+ attentive_student,
    \+ sleepy_student,
    \+ distracted_student,
    \+ phone_usage_suspected,
    \+ disengaged_student.
