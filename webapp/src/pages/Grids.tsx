import { Link } from 'react-router-dom'
import { apiUrl } from '../api'
import { Empty, Loaded, PageHead } from '../components'
import { useFetch } from '../hooks'

/** The fitted core grid drawn on every slide: the quick visual check after "Find cores". */
export default function Grids() {
  const grids = useFetch<string[]>('/api/grids')
  const picture = (name: string, width = 0) => apiUrl(`/api/grids/${encodeURIComponent(name)}${width ? `?width=${width}` : ''}`)
  return (
    <>
      <PageHead title="Core grid check" subtitle={<>Every circle should sit on a core. If a grid is shifted, press <b>Fix</b> on that slide on the <Link to="/setup/cores">Cores step of Set up</Link>.</>} />
      <Loaded state={grids} label="Loading the grid pictures…">
        {(names) => names.length ? (
          <div className="grid grid-2">
            {names.map((name) => (
              <a key={name} className="card grid-shot" href={picture(name)} target="_blank" rel="noreferrer" title="Open the picture full size">
                <img src={picture(name, 900)} alt={name} loading="lazy" />
                <div className="small secondary">{name.replace(/\.jpg$/, '').replaceAll('_', ' ')}</div>
              </a>
            ))}
          </div>
        ) : <Empty><p>No grid pictures yet: find the cores first.</p><Link className="btn primary" to="/setup/cores">Go to Cores</Link></Empty>}
      </Loaded>
    </>
  )
}
