import React, { useState, useCallback, useMemo } from 'react'
import {
  Card, Row, Col, Typography, Space, Button, Tag, DatePicker,
  Table, Progress, Empty, message, Input,
} from 'antd'
import {
  PlayCircleOutlined, ReloadOutlined, DeleteOutlined,
  EyeOutlined,
} from '@ant-design/icons'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import dayjs from 'dayjs'

import { alphaService } from '../../api/alpha'
import { useTask } from '../../hooks/useTask'
import { taskStore } from '../../stores/taskStore'
import CsvImport from './components/CsvImport'
import DataViewer, { DataStats, DataListView } from './components/DataViewer'
import type { BarDataItemWithInterval } from './components/DataViewer'

const { Text } = Typography
const { RangePicker } = DatePicker
const { TextArea } = Input

const DataPrepare: React.FC = () => {
  const queryClient = useQueryClient()

  const [symbolsText, setSymbolsText] = useState('')
  const [dateRange, setDateRange] = useState<[dayjs.Dayjs, dayjs.Dayjs]>([
    dayjs().subtract(1, 'year'),
    dayjs(),
  ])
  const [interval, setInterval] = useState<'daily' | 'minute'>('daily')
  const [taskId, setTaskId] = useState<string | null>(null)
  const [viewRecord, setViewRecord] = useState<BarDataItemWithInterval | null>(null)

  const task = useTask(taskId)

  const { data: barData, isLoading } = useQuery({
    queryKey: ['alpha-bar-data'],
    queryFn: () => alphaService.getBarData(),
  })

  const symbols = useMemo(() => {
    return symbolsText
      .split(/[\n,]+/)
      .map((s) => s.trim())
      .filter((s) => s.length > 0)
  }, [symbolsText])

  // Compute stats
  const totalSymbols = useMemo(() => {
    return (barData?.daily?.length || 0) + (barData?.minute?.length || 0)
  }, [barData])

  const totalRows = useMemo(() => {
    const dailyRows = barData?.daily?.reduce((sum, d) => sum + (d.row_count || 0), 0) || 0
    const minuteRows = barData?.minute?.reduce((sum, d) => sum + (d.row_count || 0), 0) || 0
    return dailyRows + minuteRows
  }, [barData])

  const handleDownload = useCallback(async () => {
    if (symbols.length === 0) {
      message.warning('Please enter at least one symbol')
      return
    }
    try {
      const result = await alphaService.downloadData({
        vt_symbols: symbols,
        interval,
        start: dateRange[0].format('YYYY-MM-DD'),
        end: dateRange[1].format('YYYY-MM-DD'),
      })
      setTaskId(result.task_id)
      taskStore.getState().addTask({
        id: result.task_id,
        name: `Download: ${symbols.join(', ')}`,
        status: 'running',
        progress: 0,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      })
      message.success('Download task started')
    } catch {
      message.error('Failed to start download')
    }
  }, [symbols, dateRange, interval])

  const handleDelete = useCallback(async (intervalType: string, vtSymbol: string) => {
    try {
      await alphaService.deleteBarData(intervalType, vtSymbol)
      message.success(`${vtSymbol} data deleted`)
      queryClient.invalidateQueries({ queryKey: ['alpha-bar-data'] })
    } catch {
      message.error('Delete failed')
    }
  }, [queryClient])

  const handleView = useCallback((record: BarDataItem) => {
    setViewRecord(record)
  }, [])

  const handleCloseViewer = useCallback(() => {
    setViewRecord(null)
  }, [])

  return (
    <div className="page-enter">
      <Typography.Title level={4} style={{ marginBottom: 20 }}>
        Data Prepare
      </Typography.Title>

      {/* Data Statistics Overview */}
      <DataStats
        barData={barData}
        totalSymbols={totalSymbols}
        totalRows={totalRows}
      />

      <Row gutter={[16, 16]}>
        {/* Left: Download Form */}
        <Col xs={24} lg={10}>
          <Card title="Download K-Line Data" size="small">
            <Space direction="vertical" style={{ width: '100%' }} size="middle">
              <div>
                <Text type="secondary">Symbols (one per line):</Text>
                <TextArea
                  style={{ marginTop: 8 }}
                  rows={4}
                  value={symbolsText}
                  onChange={(e) => setSymbolsText(e.target.value)}
                  placeholder="000001.SZSE&#10;600000.SSE"
                />
              </div>
              <div>
                <Text type="secondary">Date Range:</Text>
                <RangePicker
                  style={{ width: '100%', marginTop: 8 }}
                  value={dateRange}
                  onChange={(dates) => dates && setDateRange(dates as [dayjs.Dayjs, dayjs.Dayjs])}
                />
              </div>
              <div>
                <Text type="secondary">Interval:</Text>
                <Space style={{ marginTop: 8 }}>
                  <Tag.CheckableTag
                    checked={interval === 'daily'}
                    onChange={() => setInterval('daily')}
                  >
                    Daily
                  </Tag.CheckableTag>
                  <Tag.CheckableTag
                    checked={interval === 'minute'}
                    onChange={() => setInterval('minute')}
                  >
                    Minute
                  </Tag.CheckableTag>
                </Space>
              </div>
              <Button
                type="primary"
                icon={<PlayCircleOutlined />}
                onClick={handleDownload}
                loading={task?.status === 'running'}
                block
              >
                Download
              </Button>
              {task && (
                <Card size="small">
                  <Progress
                    percent={Math.round(task.progress)}
                    status={task.status === 'failed' ? 'exception' : task.status === 'completed' ? 'success' : 'active'}
                  />
                  <Text type="secondary">{task.status}</Text>
                </Card>
              )}
            </Space>
          </Card>
        </Col>

        {/* Right: Data List */}
        <Col xs={24} lg={14}>
          <Card
            title="Data List"
            size="small"
            extra={
              <Button
                type="text"
                icon={<ReloadOutlined />}
                onClick={() => queryClient.invalidateQueries({ queryKey: ['alpha-bar-data'] })}
              >
                Refresh
              </Button>
            }
          >
            {barData?.daily?.length || barData?.minute?.length ? (
              <DataListView
                barData={barData}
                onDelete={handleDelete}
                onView={handleView}
                loading={isLoading}
              />
            ) : (
              <Empty description="No data downloaded yet" />
            )}
          </Card>

          {/* CSV Import Component */}
          <CsvImport onImportComplete={() => queryClient.invalidateQueries({ queryKey: ['alpha-bar-data'] })} />
        </Col>
      </Row>

      {/* Data Viewer Modal */}
      {viewRecord && (
        <DataViewer
          record={viewRecord}
          onClose={handleCloseViewer}
        />
      )}
    </div>
  )
}

export default DataPrepare
