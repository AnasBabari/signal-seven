import React, { forwardRef, useState, useEffect } from 'react';

const loadChartEngine = () => Promise.all([import('chart.js'), import('react-chartjs-2')]);

const LazyLineChart = forwardRef(function LazyLineChart({ loadEngine = loadChartEngine, ...props }, ref) {
  const [error, setError] = useState(false);
  const [retry, setRetry] = useState(0);
  const [LineComponent, setLineComponent] = useState(null);

  useEffect(() => {
    let isMounted = true;
    setError(false);
    loadEngine()
      .then(([chartJsModule, reactChartJsModule]) => {
        if (!isMounted) return;
        const {
          Chart: ChartJS,
          CategoryScale,
          LinearScale,
          PointElement,
          LineElement,
          Title,
          Tooltip,
          Legend,
          Filler,
        } = chartJsModule;

        ChartJS.register(
          CategoryScale,
          LinearScale,
          PointElement,
          LineElement,
          Title,
          Tooltip,
          Legend,
          Filler
        );

        setLineComponent(() => reactChartJsModule.Line);
      })
      .catch(() => {
        if (isMounted) setError(true);
      });

    return () => {
      isMounted = false;
    };
  }, [retry, loadEngine]);

  if (error) return <div role="alert" className="chart-engine-error">
    Chart could not load. Prices remain available in the data table below.
    <button type="button" onClick={() => setRetry((value) => value + 1)}>Retry chart engine</button>
  </div>;

  if (!LineComponent) {
    return <div className="loading-text">Loading Chart Engine...</div>;
  }

  const Line = LineComponent;
  return <Line ref={ref} {...props} />;
});

export default LazyLineChart;
