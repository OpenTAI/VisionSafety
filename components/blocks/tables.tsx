import React, { useState, useEffect } from "react";
import type { TinaTemplate } from "tinacms";
import { PageBlocksTable } from "../../tina/__generated__/types";
import { tinaField } from "tinacms/dist/react";
import { ConfigProvider, Table } from "antd";
import playIcon from "../../assets/img/playIcon.png";
import Image from "next/image";
import { basePath } from "../util/url-helper";

interface dataType {
  key: React.Key;
  name: string;
  paper: string;
  download: number;
  datasetA: string;
  datasetB: string;
  ranking: string;
}

const columns = [
  {
    title: "模型名称",
    dataIndex: "name",
    key: "name",
    sorter: (a: dataType, b: dataType) => a.name.length - b.name.length,
  },
  {
    title: "论文",
    dataIndex: "paper",
    key: "paper",
    className: "w-96",
    render: (text: string) => (
      <div className="text-base-blue underline cursor-pointer">{text}</div>
    ),
  },
  {
    title: "模型下载量",
    dataIndex: "download",
    key: "download",
    className: "w-32",
    sorter: (a: dataType, b: dataType) => a.name.length - b.name.length,
  },
  {
    title: "对抗安全性",
    className: "!text-center",
    children: [
      {
        title: "数据集A",
        dataIndex: "datasetA",
        key: "datasetA",
        className: "!text-center",
      },
      {
        title: "数据集B",
        dataIndex: "datasetB",
        key: "datasetB",
        className: "!text-center",
      },
    ],
  },
  {
    title: "排名",
    dataIndex: "ranking",
    key: "ranking",
  },
];

export const Tables = ({
  data,
  language,
}: {
  data: PageBlocksTable;
  language: string;
}) => {
  const [activeTab, setActiveTab] = useState('黑盒');
  const [activeIndex, setActiveIndex] = useState(0);
  const [tableData, setTableData] = useState(null);

  const rowClassName = (record, index) => {
    if (index % 2 === 0) {
      return 'text-base text-sm [&>td]:!border-none'
    } else {
      return 'bg-table-blue text-base text-sm [&>td]:!border-none'
    }
  }

  const menuItems = data.modelsRanking 
  ? data.modelsRanking.map((item) => {
    return item.titleen
  }) 
  : [];

  const handleActiveIndexChange = (index) => {
    setActiveIndex(index)
    getTableData(index)
  }

  const getTableData = (index = 0) => {
    const selectedData = data.modelsRanking ? data.modelsRanking[index].rankings : [];
    const currentData = selectedData.map((item, index) => {
      return {
        key: index,
        name: item[`name${language}`],
        paper: item.paper,
        download: item.download,
        datasetA: item.datasetA,
        datasetB: item.datasetB,
        ranking: item.ranking,
      };
    });
    setTableData(currentData)
  }

  useEffect(() => {
    const params = new URL(location.href).searchParams;
    const tab = params.get("tab")
    if (tab) {
      setActiveTab(tab)
    }
    getTableData()
  }, [])

  return (
    <div>
      <div className="bg-leaderboardsBg3 bg-cover bg-center h-[392px]">
        {data[`title${language}`] && (
          <div
            className="font-extralight text-6xl w-168 text-center mx-auto pt-12 leading-none"
            data-tina-field={tinaField(data, "titleen")}
          >
            {data[`title${language}`]}
          </div>
        )}
        {data[`subtitle${language}`] && (
          <div
            className="font-light text-lg w-208 text-center mx-auto mt-4 leading-7"
            data-tina-field={tinaField(data, "subtitleen")}
          >
            {data[`subtitle${language}`]}
          </div>
        )}
        {data[`buttonText${language}`] && (
          <div
            className="bg-white/30 border-base-blue border-2 w-[331px] h-[70px] mt-4 mx-auto text-base-blue font-semibold text-2xl flex items-center justify-center"
            data-tina-field={tinaField(data, "buttonTexten")}
          >
            <Image src={playIcon} className="w-8 mr-2" alt="" />
            {data[`buttonText${language}`]}
          </div>
        )}
      </div>


      <div className="flex border w-[34rem] font-bold mx-auto mt-14">
        <div
          className={`w-1/2 p-4 text-center cursor-pointer ${activeTab === '黑盒' ? 'bg-table-blue' : ''
            }`}
          onClick={() => setActiveTab('黑盒')}
        >
          黑盒
        </div>
        <div
          className={`w-1/2 p-4 text-center cursor-pointer border-l ${activeTab === '白盒' ? 'bg-table-blue' : ''
            }`}
          onClick={() => setActiveTab('白盒')}
        >
          白盒
        </div>
      </div>

      <div className="flex justify-center mt-3 mb-24">
        <div className="w-48 bg-white pt-3 text-xs text-center font-bold border border-r-0">
          <ul>
            {menuItems.map((item, index) => (
              <li
                key={index}
                onClick={() => handleActiveIndexChange(index)}
                className={`p-2 cursor-pointer ${activeIndex === index ? 'bg-table-blue' : ''
                  }`}
              >
                {item}
              </li>
            ))}
          </ul>
        </div>
        <div className="w-[70rem] bg-table-bg border border-[#DFE3E6]">
          <ConfigProvider
            theme={{
              components: {
                Table: {
                  headerBg: "#f4faff",
                  headerColor: "#11181C",
                  headerSplitColor: "#F8F9FA",
                  cellPaddingBlock: 10,
                },
              },
            }}
          >
            <Table
              columns={columns}
              dataSource={tableData}
              pagination={false}
              rowClassName={rowClassName}
              data-tina-field={tinaField(data)}
            />
          </ConfigProvider>
          {/* <div className="py-2 px-3 bg-[#ECEEF0] w-fit mt-4 mx-auto text-[#11181C] text-base font-semibold rounded-md cursor-pointer">
            Load more
          </div> */}
        </div>
      </div>
    </div>
  );
};

export const tablesBlockSchema: TinaTemplate = {
  name: "table",
  label: "Table",
  ui: {
    previewSrc: `${basePath}/blocks/content.png`,
    defaultItem: {
      body: "Lorem ipsum dolor sit amet, consectetuer adipiscing elit. Donec odio. Quisque volutpat mattis eros. Nullam malesuada erat ut turpis. Suspendisse urna nibh, viverra non, semper suscipit, posuere a, pede.",
    },
  },
  fields: [
    {
      type: "string",
      label: "Title-En",
      name: "titleen",
    },
    {
      type: "string",
      label: "Title-Zh",
      name: "titlezh",
    },
    {
      type: "string",
      label: "Subtitle-En",
      name: "subtitleen",
    },
    {
      type: "string",
      label: "Subtitle-Zh",
      name: "subtitlezh",
    },
    {
      type: "string",
      label: "Button Text-En",
      name: "buttonTexten",
    },
    {
      type: "string",
      label: "Button Text-Zh",
      name: "buttonTextzh",
    },
    {
      type: "string",
      label: "Table Title-En",
      name: "tableTitleen",
    },
    {
      type: "string",
      label: "Table Title-Zh",
      name: "tableTitlezh",
    },
    {
      type: "object",
      label: "黑盒",
      name: "modelsRanking",
      list: true,
      ui: {
        itemProps: (item) => {
          return { label: item?.titlezh };
        },
        defaultItem: {},
      },
      fields: [
        {
          type: "string",
          label: "titlezh",
          name: "titlezh",
        },
        {
          type: "string",
          label: "titleen",
          name: "titleen",
        },
        {
          type: "object",
          label: "Rankings",
          name: "rankings",
          list: true,
          ui: {
            itemProps: (ranking) => {
              return { label: ranking?.nameen };
            },
            defaultItem: {
              nameen: "Model Name",
            },
          },
          fields: [
            {
              type: "string",
              label: "Name-En",
              name: "nameen",
            },
            {
              type: "string",
              label: "Name-Zh",
              name: "namezh",
            },
            {
              type: "string",
              label: "Paper",
              name: "paper",
            },
            {
              type: "number",
              label: "Download",
              name: "download",
            },
            {
              type: "string",
              label: "DatasetA",
              name: "datasetA",
            },
            {
              type: "string",
              label: "DatasetB",
              name: "datasetB",
            },
            {
              type: "string",
              label: "Ranking",
              name: "ranking",
            },
          ],
        },
      ],
    },
    {
      type: "object",
      label: "白盒",
      name: "modelsRanking1",
      list: true,
      ui: {
        itemProps: (item) => {
          return { label: item?.titlezh };
        },
        defaultItem: {},
      },
      fields: [
        {
          type: "string",
          label: "titlezh",
          name: "titlezh",
        },
        {
          type: "string",
          label: "titleen",
          name: "titleen",
        },
        {
          type: "object",
          label: "Rankings",
          name: "rankings",
          list: true,
          ui: {
            itemProps: (ranking) => {
              return { label: ranking?.nameen };
            },
            defaultItem: {
              nameen: "Model Name",
            },
          },
          fields: [
            {
              type: "string",
              label: "Name-En",
              name: "nameen",
            },
            {
              type: "string",
              label: "Name-Zh",
              name: "namezh",
            },
            {
              type: "string",
              label: "Paper",
              name: "paper",
            },
            {
              type: "number",
              label: "Download",
              name: "download",
            },
            {
              type: "string",
              label: "DatasetA",
              name: "datasetA",
            },
            {
              type: "string",
              label: "DatasetB",
              name: "datasetB",
            },
            {
              type: "string",
              label: "Ranking",
              name: "ranking",
            },
          ],
        },
      ],
    },
  ],
};
