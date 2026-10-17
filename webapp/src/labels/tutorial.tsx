/* The tutorial a new annotator goes through before real work (about 3–5 minutes): the two round pages on a practice
   field of the project, where nothing is saved (`practice-<round>-<stain>` sessions on the server), with a coach that
   asks for one thing at a time and waits until the person has done it. Skip at any time; "Tutorial" in the header
   shows it again. Finishing (or skipping) is remembered for the person's name. Someone new is sent here from the list
   of sessions and from a session's own link (`?next=` brings them back there afterwards). The admin can open it from
   the Labels page without a name (`?preview=1`): the same pages, nothing remembered. */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link, useLocation, useNavigate } from 'react-router-dom'
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
    { round: 2, title: 'Point at an object', text: 'Move the mouse over an outlined object. It lights up, its name shows beside it, and what detection said about it shows on the right. Every key acts on the object under the mouse.', done: ['hover'] },
    { round: 2, title: 'Give it a class', text: 'While pointing at an object, press a class number (see “Classes” on the right). Each class has its own colour, drawn solid. A dotted grey one (rejected by detection) becomes solid in that class’s colour.', done: ['class', 'subtype'] },
    { round: 2, title: 'Say which type', text: 'Some classes have types. Press the class number and, within a second, a type number (1, 2, …). Or press a type’s letter key.', done: ['subtype'] },
    { round: 2, title: 'Not sure?', text: 'Point at an object you cannot decide and press 0 (unsure). It is left out of training.', done: ['unsure'] },
    { round: 2, title: 'Not an object', text: 'Point at something that is not an object and press Delete. It turns dotted grey — it stays visible, nothing is hidden, and a class number brings it back.', done: ['remove'] },
    { round: 2, title: 'Undo', text: 'Pressed the wrong key? Press Z to undo your last change. Z takes back changes you have not saved yet.', done: ['undo'] },
    { round: 2, title: 'Bigger or smaller outline', text: 'Point at an object and press Tab for a bigger outline, Shift+Tab for a smaller one. Each object has three sizes ready, so it changes at once.', done: ['mask'] },
    { round: 2, title: 'Add a missed object', text: 'Click on tissue where detection missed an object: its outline appears as a new candidate (dotted grey with a diamond). Then press its class number to make it solid.', done: ['added'] },
    { round: 2, title: 'Precise editing', text: 'For an outline that no size fits: click the object. It is zoomed in on, and everything else is locked until you leave. (To give it a class you only point at it: no click needed.)', done: ['precise-in'] },
    { round: 2, title: 'Fix it spot by spot', text: 'Click a spot to add it, Shift+click one to leave it out, or drag one of the white dots. Z takes one step back.', done: ['precise-edit'] },
    { round: 2, title: 'Keep the edit', text: 'Click beyond the faint frame around the object (or press Enter) to keep the outline: the view goes back to where you were (Esc would cancel the edit).', done: ['precise-out'] },
    { round: 2, title: 'A clean look', text: 'Press Space to hide everything drawn over the picture and see the tissue alone. Press Space again to bring it back.', done: ['overlays'] },
    { round: 2, title: 'Mark a bad area', text: 'A fold, a smudge or empty glass? Press E and paint over it with the mouse (Shift-drag erases, scroll changes the brush). Objects in it are left out of scoring and training. Press E again when done.', done: ['exclude'] },
    { round: 2, title: 'Save your work', text: 'Nothing you do is saved yet: the panel says how many unsaved changes you have, and a small orange dot marks each object you changed. Press Save (or Ctrl+S) to keep them — you stay on this field. (In this practice, Save only pretends.)', done: ['save'] },
    { round: 2, title: 'Save and continue', text: 'When the field looks right, press “Save and continue”. Everything you left as it is counts as checked by you, the field is done, nobody else is asked to do it again, and your next field opens.', done: ['finish'] },
    { round: 1, title: 'One object at a time', text: 'The other kind of session goes through objects one at a time. If detection is right, press Enter to keep its call.', done: ['keep'] },
    { round: 1, title: 'Correct it', text: 'If detection is wrong, press the right class number, 0 for unsure or Delete for not an object. The same keys as on the field — Tab, Space and Z work here too.', done: ['class', 'subtype', 'unsure', 'remove'] },
    { round: 1, title: 'Save here too', text: 'Your answers stay on the page until you press Save (Ctrl+S). “Save and continue” finishes the field and brings the next one: objects you did not answer keep detection’s call, as your answer. If you close the page by mistake, your unsaved answers come back when you return.', done: ['save', 'finish'] },
  ]
  return typed ? steps : steps.filter((s) => s.title !== 'Say which type')
}

/** "Tutorial" in the header of the annotation pages: shows it again. */
export function TutorialLink() {
  const { session } = useSession()
  return session.annotator ? <Link className="small tutorial-link" to="/annotate/tutorial" title="Show the tutorial again">Tutorial</Link> : null
}

/** Someone new (their name has not been through the tutorial yet): to the tutorial first, before real work, and back
 *  here afterwards. */
export function useFirstTutorial() {
  const { session } = useSession()
  const navigate = useNavigate()
  const { pathname } = useLocation()
  const info = useFetch<TutorialInfo>(session.annotator ? '/api/session/tutorial' : null)
  useEffect(() => {
    if (info.data && !info.data.done && info.data.practice) {
      navigate(`/annotate/tutorial${pathname !== '/annotate' ? `?next=${encodeURIComponent(pathname)}` : ''}`, { replace: true })
    }
  }, [info.data, navigate, pathname])
}

export function Tutorial() {
  const { session } = useSession()
  const { search } = useLocation()
  const params = new URLSearchParams(search)
  const preview = params.get('preview') === '1' && session.admin
  const next = params.get('next')?.startsWith('/annotate') ? params.get('next')! : '/annotate'
  const info = useFetch<TutorialInfo>(session.annotator || preview ? '/api/session/tutorial' : null)
  if (!session.annotator && !preview) return <><PageHead title="Tutorial" subtitle="Type your name first." /><Register /></>
  return (
    <Loaded state={info} label="Opening the tutorial…">
      {(data) => data.practice ? <Coached practice={data.practice} preview={preview} next={next} />
        : <><PageHead title="Tutorial" subtitle="Nothing to practise on yet." />
          <Empty><p style={{ marginTop: 0 }}>The practice field appears once detection has proposed objects on this project’s images (the v2 detection’s
            “Find candidates” and “Decide” steps on the Run page).</p>
            <Link className="btn" to={preview ? '/' : '/annotate'}>{preview ? 'Back' : 'All sessions'}</Link></Empty></>}
    </Loaded>
  )
}

function Coached({ practice, preview, next }: { practice: NonNullable<TutorialInfo['practice']>; preview: boolean; next: string }) {
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
    if (preview) return  // the admin looking: nothing is remembered
    try { await api.post('/api/session/tutorial', { done: true }) } catch (e) { setError((e as Error).message) }
  }, [preview])
  useEffect(() => { if (finished) void remember() }, [finished, remember])
  const at = useRef(0)
  at.current = index
  const onEvent = useCallback((event: RoundEvent) => {
    const i = at.current
    if (i < steps.length && steps[i].done.includes(event)) { setIndex(i + 1); setCheer(true) }
  }, [steps])
  useEffect(() => { if (!cheer) return; const timer = setTimeout(() => setCheer(false), 1200); return () => clearTimeout(timer) }, [cheer])
  const leave = () => { if (preview) window.history.length > 1 ? window.history.back() : window.close(); else navigate(next) }
  const skip = async () => { await remember(); leave() }

  const round = finished ? 1 : step.round
  const detail = round === 2 ? field : stream
  const coach = (
    <aside className={`card tutorial-coach${round === 2 ? ' in-panel' : ''}`} role="dialog" aria-label="Tutorial">
      <div className="row small"><b>Tutorial</b><span className="muted">{preview ? 'preview — what annotators see; nothing is saved' : 'practice — nothing here is saved'}</span>
        <span className="spacer" />
        {!finished && <button className="link-btn small" onClick={skip}>{preview ? 'Close' : 'Skip tutorial'}</button>}</div>
      {finished ? <>
        <h3>You are ready</h3>
        <p className="small secondary">That is all: point, press a key, correct detection’s work, and save each field. You can see this again with “Tutorial” at the top.</p>
        {preview ? <button className="btn primary" onClick={leave}>Close the preview</button> : <Link className="btn primary" to={next}>Start labelling →</Link>}
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
  )
  return (
    <div className="tutorial">
      {!detail.data ? <Loading label="Opening the practice field…" /> : round === 2
        ? <FixField key="practice-field" detail={detail.data} reload={() => undefined} tile={practice.tile_id} onEvent={onEvent} coach={coach}
          onFinished={() => setIndex((i) => Math.max(i, steps.findIndex((s) => s.round === 1)))} />
        : <CheckCalls key="practice-stream" detail={detail.data} onEvent={onEvent} />}
      {round === 1 && coach}
    </div>
  )
}
