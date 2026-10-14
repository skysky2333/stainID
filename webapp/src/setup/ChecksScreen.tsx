import { useState } from 'react'
import type { ReactElement } from 'react'
import type { Fix, Issue } from '../api'
import { api } from '../api'
import { ErrorNote, withError } from '../components'
import { useFetch } from '../hooks'
import { Advanced, Found, ScreenFoot } from './parts'
import type { ScreenId, ScreenProps } from './Setup'

const LEVEL = { problem: 'Needs fixing', warning: 'Please look', note: 'For information' }

export default function ChecksScreen({ next, back, go, onChanged }: ScreenProps) {
  const issues = useFetch<Issue[]>('/api/setup/checks')
  const [message, setMessage] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState('')
  const apply = (fix: Fix) => {
    if (fix.action === 'goto') { go(fix.args.screen as ScreenId); return }
    withError(async () => {
      setBusy(fix.label + JSON.stringify(fix.args))
      try {
        const result = await api.post<{ message: string }>('/api/setup/checks/fix', fix)
        setMessage(result.message + '.'); issues.reload(); onChanged()
      } finally { setBusy('') }
    }, setError)
  }
  const list = issues.data ?? []
  const problems = list.filter((i) => i.level === 'problem')
  const shown = list.filter((i) => i.level !== 'note')
  const notes = list.filter((i) => i.level === 'note')
  const fixButton = (fix: Fix, primary = false) => (
    <button key={fix.label + JSON.stringify(fix.args)} className={`btn small${primary ? ' primary' : ''}`} disabled={Boolean(busy)} onClick={() => apply(fix)}>
      {busy === fix.label + JSON.stringify(fix.args) ? 'Working…' : fix.label}</button>
  )

  return (
    <div className="screen">
      {issues.data && (shown.length === 0 ? <Found>Everything fits together: every core position is on a map, and the map and the donor table agree.</Found>
        : <Found tone="warn">{problems.length ? `${problems.length} ${problems.length === 1 ? 'thing needs' : 'things need'} fixing before the analysis can run` : 'Nothing stops the analysis'}
          {shown.length > problems.length ? `, ${shown.length - problems.length} to look at` : ''}. Each comes with a fix where the answer is clear.</Found>)}
      {message && <p className="small status-finished">{message}</p>}
      <ErrorNote error={error ?? issues.error} />
      {shown.map((issue) => <IssueCard key={issue.id} issue={issue} fixButton={fixButton} />)}
      {notes.length > 0 && (
        <Advanced title={`${notes.length} ${notes.length === 1 ? 'note' : 'notes'}`} hint="for information, nothing to fix">
          {notes.map((issue) => <IssueCard key={issue.id} issue={issue} fixButton={fixButton} />)}
        </Advanced>
      )}
      <ScreenFoot back={back} next={next} nextLabel={problems.length ? 'Next (fix the problems first)' : 'Next: run'} />
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
          {issue.more > 0 && <li className="muted">… and {issue.more} more</li>}
        </ul>
      )}
      {issue.fixes.length > 0 && <div className="row">{issue.fixes.map((f, i) => fixButton(f, i === 0))}</div>}
    </div>
  )
}
