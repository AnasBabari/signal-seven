import React, { forwardRef, useState, useEffect, useRef } from 'react';
import { withDeadline } from '../api/request';

const loadChartEngine = () => import('./chart-engine.js');

const LazyLineChart = forwardRef(function LazyLineChart({ loadEngine = loadChartEngine, onErrorChange = null, ...props }, ref) {
  const [error, setError] = useState(false);
  const [retry, setRetry] = useState(0);
  const [LineComponent, setLineComponent] = useState(null);
  const onErrorChangeRef = useRef(onErrorChange);
  onErrorChangeRef.current = onErrorChange;

  useEffect(() => {
    let isMounted = true;
    setError(false);
    onErrorChangeRef.current?.(false);
    withDeadline(() => loadEngine(), { timeoutMs: 20_000 })
      .then((engineModule) => {
        if (!isMounted) return;
        const Line = engineModule?.Line;
        if (!Line) {
          throw new Error('Chart engine module did not export Line.');
        }
        setLineComponent(() => Line);
      })
      .catch(() => {
        if (isMounted) {
          setError(true);
          onErrorChangeRef.current?.(true);
        }
      });

    return () => {
      isMounted = false;
    };
  }, [retry, loadEngine]);

  if (error) return (
    <div role="alert" className="chart-engine-error">
      Chart could not load. Prices remain available in the data table below.
      <button type="button" className="retry-action-btn" onClick={() => setRetry((value) => value + 1)}>
        Retry chart engine
      </button>
    </div>
  );

  if (!LineComponent) {
    return <div className="loading-text">Loading Chart Engine...</div>;
  }

  const Line = LineComponent;
  return <Line ref={ref} {...props} />;
});

export default LazyLineChart;
