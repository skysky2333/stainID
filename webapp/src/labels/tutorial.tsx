/* The tutorial a new annotator goes through before real work (about 3–5 minutes): the two round pages on a practice
   field of the project, where nothing is saved (`practice-<round>-<stain>` sessions on the server), with a coach that
   asks for one thing at a time and waits until the person has done it. Skip at any time; "Tutorial" in the header
   shows it again. Finishing (or skipping) is remembered for the person's name. Someone new is sent here from the list
   of sessions and from a session's own link (`?next=` brings them back there afterwards). The admin can open it from
   the Labels page without a name (`?preview=1`): the same pages, nothing remembered. `?kind=areas` is the short tutorial
   of "Mark bad areas" (labels/markareas.tsx) on a practice core (`practice-areas`), remembered in this browser. */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import type { SessionDetail } from '../api'
import { api } from '../api'
import { Empty, ErrorNote, Loaded, Loading, PageHead } from '../components'
import { useFetch } from '../hooks'
import { CheckCalls } from './checkcalls'
import { FixField } from './fixfield'
import type { AreaEvent } from './markareas'
import { MarkAreas, PRACTICE_AREAS } from './markareas'
import type { RoundEvent } from './rounds'
import { Register, useSession } from '../session'

interface TutorialInfo { done: boolean; practice: { stain: string; tile_id: string; round1: string; round2: string } | null }
/** One step: done once the page reports one of `done` (`times` of them: "reject a few"). */
interface Step { round: 1 | 2; title: string; text: string; done: RoundEvent[]; times?: number }

/** The steps, building up: reject what detection got wrong (Del), bring types back with a number, change a type (on a
 *  stain with types), unsure (`), a clean look (Space), precise editing, the brush (B), saving. `keys`: the practice
 *  stain's key line (`Vocabulary.keymap` summary); `typed`: its objects have several types to choose from. */
function stepsFor(typed: boolean, keys: string, counted: string): Step[] {
  const others = 'On 6E10 and AT8, 1–4 pick the type (compact, diffuse… or pretangle, mature tangle…) and Q W E the named look-alikes (CAA, a thread…).'
  const steps: Step[] = [
    { round: 2, title: 'Point at an object', text: 'Move the mouse over an outlined object. It lights up, its name shows beside it, and what detection said about it shows on the right. Every key acts on the object under the mouse.', done: ['hover'] },
    { round: 2, title: 'Reject what detection got wrong', text: 'Find a few objects detection got wrong, point at each and press Del. They turn dotted grey and stay visible. Some objects are already dotted grey: detection rejected those itself, and they look just the same.', done: ['remove'], times: 2 },
    { round: 2, title: 'Bring the type back', text: `Point at a dotted grey object that is a real one and press ${counted}: it becomes solid in that type’s colour. Each type has its own colour.${typed ? '' : ` ${others}`}`, done: ['class'] },
    { round: 2, title: 'Change the type', text: `Point at a solid object and press another number: it changes to that type and colour. ${others}`, done: ['subtype'] },
    { round: 2, title: 'Not sure?', text: 'Point at an object you cannot decide and press ` (the key left of 1; 0 works too). It turns dotted amber with a “?” and is left out of training and evaluation.', done: ['unsure'] },
    { round: 2, title: 'Undo', text: 'Pressed the wrong key? Press Z to undo your last change. Z takes back changes you have not saved yet.', done: ['undo'] },
    { round: 2, title: 'A clean look', text: 'Press Space to hide everything drawn over the picture and see the tissue alone. Press Space again to bring it back.', done: ['overlays'] },
    { round: 2, title: 'Precise editing', text: 'For an outline that is not right: click the object. It is zoomed in on, and everything else is locked until you leave. (To change its type you only point at it: no click needed.)', done: ['precise-in'] },
    { round: 2, title: 'Fix it spot by spot', text: 'Click a spot to add it, Shift+click one to leave it out, or drag one of the white dots. Z takes one step back.', done: ['precise-edit'] },
    { round: 2, title: 'Leave precise editing', text: 'Click outside the faint frame around the object (or press Enter) to keep the outline: the view goes back to where you were (Esc would cancel the edit).', done: ['precise-out'] },
    { round: 2, title: 'Bigger or smaller outline', text: 'Point at an object and press Tab for a bigger outline, Shift+Tab for a smaller one. Each object has three sizes ready, so it changes at once.', done: ['mask'] },
    { round: 2, title: 'Add a missed object', text: `Click on tissue where detection missed an object: its outline appears as a new candidate (dotted grey with a diamond). Then press ${counted} to make it solid.`, done: ['added'] },
    { round: 2, title: 'Mark a bad area', text: 'Hatched areas are what the analysis leaves out: folds, artifacts, empty glass, found by detection. If you see one it missed, press B for the brush. While it is on, 1–4 choose the area’s type (fold, artifact, no tissue, out of focus) and the class keys rest. Paint over it; Shift-drag marks a part fine, also part of a detected one. Your areas teach and test the bad-area detection; the analysis keeps detection’s own. Press B again when done.', done: ['exclude'] },
    { round: 2, title: 'Save your work', text: 'Nothing you do is saved yet: the panel says how many unsaved changes you have, and a small orange dot marks each object you changed. Press Save (or Ctrl+S) to keep them — you stay on this field. (In this practice, Save only pretends.)', done: ['save'] },
    { round: 2, title: 'Save and continue', text: 'When the field looks right, press “Save and continue”. Everything you left as it is counts as checked by you, the field is done, nobody else is asked to do it again, and your next field opens.', done: ['finish'] },
    { round: 1, title: 'One object at a time', text: 'The other kind of session goes through objects one at a time. If detection is right, press Enter to keep its call.', done: ['keep'] },
    { round: 1, title: 'Correct it', text: `If detection is wrong, press the right key — ${keys.replace(/ · B brush$/, '')}. The same keys as on the field; Tab, Space and Z work here too.`, done: ['class', 'subtype', 'unsure', 'remove'] },
    { round: 1, title: 'Save here too', text: 'Your answers stay on the page until you press Save (Ctrl+S). “Save and continue” finishes the field and brings the next one: objects you did not answer keep detection’s call, as your answer. If you close the page by mistake, your unsaved answers come back when you return.', done: ['save', 'finish'] },
  ]
  return typed ? steps : steps.filter((s) => s.title !== 'Change the type')
}

/** "Tutorial" in the header of the annotation pages: shows it again. */
export function TutorialLink() {
  const { session } = useSession()
  return session.annotator ? <Link className="small tutorial-link" to="/annotate/tutorial" title="Show the tutorial again">Tutorial</Link> : null
}

const areasDoneKey = (person: string | null | undefined) => `stainid.tutorial.areas.${person ?? ''}`
function areasDone(person: string | null | undefined) {
  try { return localStorage.getItem(areasDoneKey(person)) === '1' } catch { return false }
}

/** Someone new (their name has not been through the tutorial yet): to the tutorial first, before real work, and back
 *  here afterwards. `kind`: what the page is (`null` while not known yet); a "Mark bad areas" session sends them to its
 *  own short tutorial, once in this browser. */
export function useFirstTutorial(kind: 'objects' | 'areas' | null = 'objects') {
  const { session } = useSession()
  const navigate = useNavigate()
  const { pathname } = useLocation()
  const info = useFetch<TutorialInfo>(session.annotator && kind === 'objects' ? '/api/session/tutorial' : null)
  useEffect(() => {
    if (kind === 'areas' && session.annotator && !areasDone(session.annotator)) {
      navigate(`/annotate/tutorial?kind=areas&next=${encodeURIComponent(pathname)}`, { replace: true })
    } else if (kind === 'objects' && info.data && !info.data.done && info.data.practice) {
      navigate(`/annotate/tutorial${pathname !== '/annotate' ? `?next=${encodeURIComponent(pathname)}` : ''}`, { replace: true })
    }
  }, [kind, info.data, navigate, pathname, session.annotator])
}

export function Tutorial() {
  const { session } = useSession()
  const { search } = useLocation()
  const params = new URLSearchParams(search)
  const preview = params.get('preview') === '1' && session.admin
  const next = params.get('next')?.startsWith('/annotate') ? params.get('next')! : '/annotate'
  const areas = params.get('kind') === 'areas'
  const info = useFetch<TutorialInfo>((session.annotator || preview) && !areas ? '/api/session/tutorial' : null)
  if (!session.annotator && !preview) return <><PageHead title="Tutorial" subtitle="Type your name first." /><Register /></>
  if (areas) return <AreasCoached preview={preview} next={next} />
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
  const keys = field.data?.meta.keys
  const counted = keys?.groups?.find((g) => g.group === 'counted')
  const typed = (counted?.items.length ?? 0) > 1
  const steps = useMemo(() => stepsFor(typed, keys?.summary ?? '', counted ? (counted.items.length > 1 ? `a number (${counted.keys})` : counted.keys) : '1'),
    [typed, keys?.summary, counted])
  const count = useRef(0)  // events of the step so far (for steps done after several)
  const [index, setIndex] = useState(0)
  const [cheer, setCheer] = useState(false)
  useEffect(() => { count.current = 0 }, [index])
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
    if (i >= steps.length || !steps[i].done.includes(event)) return
    count.current += 1
    if (count.current < (steps[i].times ?? 1)) return
    count.current = 0
    setIndex(i + 1); setCheer(true)
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

/* ---------- the tutorial of "Mark bad areas" ---------- */

interface AreaStep { title: string; text: string; done: AreaEvent[] }
const AREA_STEPS: AreaStep[] = [
  { title: 'One core, all its stains', text: 'You see one core at a time with each of its stains (the tabs above the picture). Hatched parts are what detection itself leaves out of the analysis; they do not change here. Press Tab or ] (or click a tab) to look at the next stain, Shift+Tab or [ to go back.', done: ['stain'] },
  { title: 'Choose a type', text: 'Press 1–4 to choose what is wrong: 1 fold or tissue damage, 2 artifact (scratch, bubble, debris), 3 no tissue, 4 out of focus or bad staining. The brush on the picture takes its colour.', done: ['type'] },
  { title: 'Paint a bad area', text: 'Drag over a part that should not be counted. It is hatched in the type’s colour. The middle mouse button moves the picture; Ctrl+wheel or + and − zoom.', done: ['paint'] },
  { title: 'Brush size', text: 'Turn the mouse wheel over the picture to make the brush bigger or smaller.', done: ['size'] },
  { title: 'Marked fine', text: 'Hold Shift and drag over a part that is fine after all, for example part of an area detection hatched. It shows faint green.', done: ['clear'] },
  { title: 'Undo', text: 'Press Z to take back your last stroke.', done: ['undo'] },
  { title: 'A clean look', text: 'Press Space to hide everything drawn over the tissue, and Space again to bring it back.', done: ['overlays'] },
  { title: 'This stain is done', text: 'When a stain looks right — painted, or fine as it is — press Enter. It gets a tick and the next stain opens.', done: ['fine'] },
  { title: 'Save the core', text: 'Press Enter on the last stain (or “Save and continue”) once every stain has a tick: the core is saved and the next one opens. (In this practice nothing is kept.)', done: ['saved'] },
]

function AreasCoached({ preview, next }: { preview: boolean; next: string }) {
  const navigate = useNavigate()
  const { session } = useSession()
  const detail = useFetch<SessionDetail>(`/api/annotate/${PRACTICE_AREAS}`)
  const [index, setIndex] = useState(0)
  const [cheer, setCheer] = useState(false)
  const step = AREA_STEPS[Math.min(index, AREA_STEPS.length - 1)]
  const finished = index >= AREA_STEPS.length
  const remember = useCallback(() => {
    if (preview) return  // the admin looking: nothing is remembered
    try { localStorage.setItem(areasDoneKey(session.annotator), '1') } catch { /* a private window: shown again next time */ }
  }, [preview, session.annotator])
  useEffect(() => { if (finished) remember() }, [finished, remember])
  const at = useRef(0)
  at.current = index
  const onEvent = useCallback((event: AreaEvent) => {
    const i = at.current
    if (i < AREA_STEPS.length && AREA_STEPS[i].done.includes(event)) { setIndex(i + 1); setCheer(true) }
  }, [])
  useEffect(() => { if (!cheer) return; const timer = setTimeout(() => setCheer(false), 1200); return () => clearTimeout(timer) }, [cheer])
  const leave = () => { if (preview) window.history.length > 1 ? window.history.back() : window.close(); else navigate(next) }
  const coach = (
    <aside className="card tutorial-coach in-panel" role="dialog" aria-label="Tutorial">
      <div className="row small"><b>Tutorial</b><span className="muted">{preview ? 'preview — what annotators see; nothing is saved' : 'practice — nothing here is saved'}</span>
        <span className="spacer" />
        {!finished && <button className="link-btn small" onClick={() => { remember(); leave() }}>{preview ? 'Close' : 'Skip tutorial'}</button>}</div>
      {finished ? <>
        <h3>You are ready</h3>
        <p className="small secondary">Look at each stain, paint what should not be counted, press Enter for each stain, and the core is saved. The areas you mark
          teach and test the bad-area detection; the analysis uses detection’s own areas.</p>
        {preview ? <button className="btn primary" onClick={leave}>Close the preview</button> : <Link className="btn primary" to={next}>Start →</Link>}
      </> : <>
        <div className="tutorial-step small muted">Step {index + 1} of {AREA_STEPS.length}{cheer && <span className="good-text"> · well done ✓</span>}</div>
        <h3>{step.title}</h3>
        <p className="small">{step.text}</p>
        {index === 0 && <p className="small secondary">The areas you mark teach and test the bad-area detection; the analysis uses detection’s own areas.</p>}
        <div className="tutorial-dots" aria-hidden="true">{AREA_STEPS.map((s, i) => <span key={s.title} className={i < index ? 'done' : i === index ? 'now' : ''} />)}</div>
        <div className="row">
          <button className="btn small" disabled={index === 0} onClick={() => setIndex(index - 1)}>Back</button>
          <button className="link-btn small" onClick={() => setIndex(index + 1)}>I did it / next</button>
        </div>
      </>}
    </aside>
  )
  return (
    <div className="tutorial">
      {detail.error ? <><PageHead title="Tutorial" subtitle="Nothing to practise on yet." />
        <Empty><p style={{ marginTop: 0 }}>The practice core appears once the project has analysed cores.</p><ErrorNote error={detail.error} />
          <Link className="btn" to={preview ? '/' : '/annotate'}>{preview ? 'Back' : 'All sessions'}</Link></Empty></>
        : !detail.data ? <Loading label="Opening the practice core…" />
          : <MarkAreas detail={detail.data} onEvent={onEvent} coach={coach} onFinished={() => setIndex((i) => Math.max(i, AREA_STEPS.length))} />}
    </div>
  )
}
