/* The tutorial a new annotator goes through before real work (about 3–5 minutes): the two round pages on a practice
   field of the project, where nothing is saved (`practice-<round>-<stain>` sessions on the server), with a coach that
   asks for one thing at a time and waits until the person has done it. Skip at any time; "Tutorial" in the header
   shows it again. Finishing (or skipping) is remembered for the person's name. */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import type { SessionDetail } from '../api'
import { api } from '../api'
import { Empty, ErrorNote, Loaded, Loading, PageHead } from '../components'
import { useFetch } from '../hooks'
import { CheckCalls } from './checkcalls'
import { FixField } from './fixfield'
import type { RoundEvent } from './rounds'
import { Register, useSession } from '../session'

interface TutorialInfo { done: boolean; practice: { stain: string; tile_id: string; round1: string; round2: string } | null }
interface Step { round: 1 | 2; title: string; text: string; done: RoundEvent[] }

function stepsFor(typed: boolean): Step[] {
  const steps: Step[] = [
    { round: 2, title: 'Point at an object', text: 'Move the mouse over an outlined object. It lights up, and what the AI said about it shows on the right. Every key acts on the object under the mouse.', done: ['hover'] },
    { round: 2, title: 'Give it a class', text: 'While pointing at an object, press a class number (see “Colours and keys”). A dotted grey one becomes solid in that class’s colour.', done: ['class', 'subtype'] },
    { round: 2, title: 'Say which type', text: 'Some classes have types. Press the class number and, within a second, a type number (1, 2, …). Or press a type’s letter key.', done: ['subtype'] },
    { round: 2, title: 'Not sure?', text: 'Point at an object you cannot decide and press 0 (unsure). It is left out of training.', done: ['unsure'] },
    { round: 2, title: 'Not an object', text: 'Point at something that is not an object and press Delete. It turns dotted grey — it stays visible, nothing is hidden.', done: ['remove'] },
    { round: 2, title: 'Undo', text: 'Pressed the wrong key? Press Z to undo your last change. Z takes back changes you have not saved yet.', done: ['undo'] },
    { round: 2, title: 'Bigger or smaller outline', text: 'Point at an object and press Tab for a bigger outline, Shift+Tab for a smaller one. Each object has three sizes ready, so it changes at once.', done: ['mask'] },
    { round: 2, title: 'Add a missed object', text: 'Click on tissue where the AI missed an object: its outline appears (dotted, “new”). Then press its class number to keep it.', done: ['added'] },
    { round: 2, title: 'Precise editing', text: 'For an outline that no size fits: point at the object and press Space. Everything else is locked.', done: ['precise-in'] },
    { round: 2, title: 'Fix it spot by spot', text: 'Right-click a spot to include it, left-click one to leave it out, or drag one of the white dots.', done: ['precise-edit'] },
    { round: 2, title: 'Keep the edit', text: 'Press Space again to keep the outline (Esc would cancel it).', done: ['precise-out'] },
    { round: 2, title: 'Save your work', text: 'Nothing you do is saved yet: the count at the top says how many unsaved changes you have, and a small orange dot marks each object you changed. Press Save (or Ctrl+S) to keep them — you stay on this field. (In this practice, Save only pretends.)', done: ['save'] },
    { round: 2, title: 'Save and continue', text: 'When the field looks right, press “Save and continue”. Everything you left as it is counts as checked by you, the field is done, nobody else is asked to do it again, and your next field opens.', done: ['finish'] },
    { round: 1, title: 'One object at a time', text: 'The other kind of session goes through a field one object at a time. If the AI is right, press Enter to keep its call.', done: ['keep'] },
    { round: 1, title: 'Correct it', text: 'If the AI is wrong, press the right class number, 0 for unsure or Delete for not an object. The same keys as on the field — Tab, Space and Z work here too.', done: ['class', 'subtype', 'unsure', 'remove'] },
    { round: 1, title: 'Save here too', text: 'Your answers stay on the page until you press Save (Ctrl+S). Once every object of the field has an answer, “Save and continue” finishes the field and brings the next one. If you close the page by mistake, your unsaved answers come back when you return.', done: ['save', 'finish'] },
  ]
  return typed ? steps : steps.filter((s) => s.title !== 'Say which type')
}

/** "Tutorial" in the header of the annotation pages: shows it again. */
export function TutorialLink() {
  const { session } = useSession()
  return session.annotator ? <Link className="small tutorial-link" to="/annotate/tutorial" title="Show the tutorial again">Tutorial</Link> : null
}

/** Someone new (their name has not been through the tutorial yet): to the tutorial first, before real work. */
export function useFirstTutorial() {
  const { session } = useSession()
  const navigate = useNavigate()
  const info = useFetch<TutorialInfo>(session.annotator ? '/api/session/tutorial' : null)
  useEffect(() => {
    if (info.data && !info.data.done && info.data.practice) navigate('/annotate/tutorial', { replace: true })
  }, [info.data, navigate])
}

export function Tutorial() {
  const { session } = useSession()
  const info = useFetch<TutorialInfo>(session.annotator ? '/api/session/tutorial' : null)
  if (!session.annotator) return <><PageHead title="Tutorial" subtitle="Type your name first." /><Register /></>
  return (
    <Loaded state={info} label="Opening the tutorial…">
      {(data) => data.practice ? <Coached practice={data.practice} />
        : <><PageHead title="Tutorial" subtitle="Nothing to practise on yet." />
          <Empty><p style={{ marginTop: 0 }}>The practice field appears once the AI has proposed objects on this project’s images.</p>
            <Link className="btn" to="/annotate">All sessions</Link></Empty></>}
    </Loaded>
  )
}

function Coached({ practice }: { practice: NonNullable<TutorialInfo['practice']> }) {
  const navigate = useNavigate()
  const field = useFetch<SessionDetail>(`/api/annotate/${practice.round2}`)
  const stream = useFetch<SessionDetail>(`/api/annotate/${practice.round1}`)
  const typed = !!field.data?.meta.vocabulary.classes.some((c) => !c.hidden && c.subtypes.some((s) => !s.hidden))
  const steps = useMemo(() => stepsFor(typed), [typed])
  const [index, setIndex] = useState(0)
  const [cheer, setCheer] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const step = steps[Math.min(index, steps.length - 1)]
  const finished = index >= steps.length

  const remember = useCallback(async () => {
    try { await api.post('/api/session/tutorial', { done: true }) } catch (e) { setError((e as Error).message) }
  }, [])
  useEffect(() => { if (finished) void remember() }, [finished, remember])
  const at = useRef(0)
  at.current = index
  const onEvent = useCallback((event: RoundEvent) => {
    const i = at.current
    if (i < steps.length && steps[i].done.includes(event)) { setIndex(i + 1); setCheer(true) }
  }, [steps])
  useEffect(() => { if (!cheer) return; const timer = setTimeout(() => setCheer(false), 1200); return () => clearTimeout(timer) }, [cheer])
  const skip = async () => { await remember(); navigate('/annotate') }

  const round = finished ? 1 : step.round
  const detail = round === 2 ? field : stream
  return (
    <div className="tutorial">
      {!detail.data ? <Loading label="Opening the practice field…" /> : round === 2
        ? <FixField key="practice-field" detail={detail.data} reload={() => undefined} tile={practice.tile_id} onEvent={onEvent}
          onFinished={() => setIndex((i) => Math.max(i, steps.findIndex((s) => s.round === 1)))} />
        : <CheckCalls key="practice-stream" detail={detail.data} onEvent={onEvent} />}
      <aside className="card tutorial-coach" role="dialog" aria-label="Tutorial">
        <div className="row small"><b>Tutorial</b><span className="muted">practice — nothing here is saved</span><span className="spacer" />
          {!finished && <button className="link-btn small" onClick={skip}>Skip tutorial</button>}</div>
        {finished ? <>
          <h3>You are ready</h3>
          <p className="small secondary">That is all: point, press a key, correct the AI’s work, and save each field. You can see this again with “Tutorial” at the top.</p>
          <Link className="btn primary" to="/annotate">Start labelling →</Link>
        </> : <>
          <div className="tutorial-step small muted">Step {index + 1} of {steps.length}{cheer && <span className="good-text"> · well done ✓</span>}</div>
          <h3>{step.title}</h3>
          <p className="small">{step.text}</p>
          <div className="tutorial-dots" aria-hidden="true">{steps.map((s, i) => <span key={s.title} className={i < index ? 'done' : i === index ? 'now' : ''} />)}</div>
          <div className="row">
            <button className="btn small" disabled={index === 0} onClick={() => setIndex(index - 1)}>Back</button>
            <button className="link-btn small" onClick={() => setIndex(index + 1)}>I did it / next</button>
          </div>
        </>}
        <ErrorNote error={error ?? field.error ?? stream.error} />
      </aside>
    </div>
  )
}
