import { useState } from 'react'
import type { ReactElement } from 'react'
import type { Fix, Issue } from '../api'
import { api } from '../api'
import { ErrorNote, withError } from '../components'
import { useFetch } from '../hooks'
import { Advanced, Found, plural, ScreenData, ScreenFoot } from './parts'
import type { ScreenId, ScreenProps } from './Setup'

const LEVEL = { problem: 'Needs fixing', warning: 'Please look', note: 'For information' }
const fixKey = (fix: Fix) => fix.label + JSON.stringify(fix.args)

export default function ChecksScreen(props: ScreenProps) {
  const issues = useFetch<Issue[]>('/api/setup/checks')
  return <ScreenData state={issues} label="Comparing what was set up…">{(list) => <Checks {...props} list={list} reload={issues.reload} />}</ScreenData>
}

function Checks({ next, back, nextLabel, go, onChanged, overview, list, reload }: ScreenProps & { list: Issue[]; reload: () => void }) {
  const tma = (overview?.kind ?? 'tma') === 'tma'
  const [message, setMessage] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState('')
  const apply = (fix: Fix) => {
    if (fix.action === 'goto') { go(fix.args.screen as ScreenId); return }
    withError(async () => {
      setBusy(fixKey(fix)); setMessage('')
      try {
        const result = await api.post<{ message: string }>('/api/setup/checks/fix', fix)
        setMessage(result.message + '.'); reload(); onChanged()
      } finally { setBusy('') }
    }, setError)
  }
  const problems = list.filter((i) => i.level === 'problem')
  const shown = list.filter((i) => i.level !== 'note')
  const notes = list.filter((i) => i.level === 'note')
  const fixButton = (fix: Fix, primary = false) => (
    <button key={fixKey(fix)} className={`btn small${primary ? ' primary' : ''}`} disabled={Boolean(busy)} onClick={() => apply(fix)}>
      {busy === fixKey(fix) ? 'Working…' : fix.label}</button>
  )

  return (
    <div className="screen">
      {shown.length === 0 ? <Found>{tma ? 'Everything fits together: every core position is on a map, and the maps and the donor table agree.' : 'Everything fits together: every image has a donor, and the images and the donor table agree.'}</Found>
        : <Found tone="warn"><b>{problems.length ? `${plural(problems.length, 'thing needs', 'things need')} fixing before the analysis can run` : 'Nothing stops the analysis'}</b>
          {shown.length > problems.length ? `, ${plural(shown.length - problems.length, 'thing')} to look at` : ''}. Each comes with a fix where the answer is clear.</Found>}
      {message && <p className="small status-finished" role="status">{message}</p>}
      <ErrorNote error={error} />
      {shown.map((issue) => <IssueCard key={issue.id} issue={issue} fixButton={fixButton} />)}
      {notes.length > 0 && (
        <Advanced title={plural(notes.length, 'note')} hint="for information, nothing to fix">
          {notes.map((issue) => <IssueCard key={issue.id} issue={issue} fixButton={fixButton} />)}
        </Advanced>
      )}
      <div className="row"><button className="link-btn small" onClick={reload}>Check again</button></div>
      <ScreenFoot back={back} next={next} nextLabel={nextLabel} quiet={problems.length > 0}
        extra={problems.length > 0 && <span className="small cell-note">Fix the problems first, or the analysis stops at them</span>} />
    </div>
  )
}

function IssueCard({ issue, fixButton }: { issue: Issue; fixButton: (fix: Fix, primary?: boolean) => ReactElement }) {
  return (
    <div className={`card issue issue-${issue.level}`}>
      <div className="row"><span className={`badge level-${issue.level}`}>{LEVEL[issue.level]}</span><b>{issue.title}</b></div>
      {issue.detail && <p className="small secondary" style={{ margin: '4px 0' }}>{issue.detail}</p>}
      {issue.items.length > 0 && (
        <ul className="issue-items">
          {issue.items.map((item) => <li key={item.text}><span>{item.text}</span>{item.fixes.length > 0 && <span className="row">{item.fixes.map((f) => fixButton(f))}</span>}</li>)}
          {issue.more > 0 && <li className="muted">… and {issue.more.toLocaleString()} more</li>}
        </ul>
      )}
      {issue.fixes.length > 0 && <div className="row">{issue.fixes.map((f, i) => fixButton(f, i === 0 && issue.level === 'problem'))}</div>}
    </div>
  )
}
