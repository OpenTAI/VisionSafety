// import { iconSchema } from "../util/icon";
import {
  PageBlocksTests,
  PageBlocksTestsItems,
} from "../../tina/__generated__/types";
import { tinaField } from "tinacms/dist/react";
import { basePath } from "../util/url-helper";

export const TestsItem = ({ data }: { data: PageBlocksTestsItems }) => {
  const itemData = data as Record<string, any>;
  const nameZh = itemData?.namezh || itemData?.name;
  const nameEn = itemData?.nameen;
  // unified blue color style for all cells
  const borderCls = "border-blue-200";
  const badgeTextCls = "text-blue-700";

  return (
    <div
      className={`m-2 w-60 sm:w-64 rounded-lg border ${borderCls} hover:bg-blue-50/60 backdrop-blur shadow-sm hover:shadow-lg transition duration-200 hover:scale-[1.02]`}
    >
      <div className="px-5 py-4 flex flex-col items-center">
        <div className="w-full grid gap-1.5 text-center">
          {nameZh ? (
            <div
              className="text-base font-semibold text-gray-900"
              data-tina-field={tinaField(itemData, "namezh")}
            >
              {nameZh}
            </div>
          ) : null}
          {nameEn ? (
            <span
              className={`mt-1 inline-flex items-center justify-center text-center rounded-full border ${borderCls} px-3 py-1.5 text-[11px] font-medium ${badgeTextCls}`}
            >
              {nameEn || nameZh}
            </span>
          ) : null}
        </div>
      </div>
    </div>
  );
};

export const Tests = ({
  data,
  language,
}: {
  data: PageBlocksTests;
  language: string;
}) => {
  return (
    <div className="w-full bg-bg-greyB">
      <div className="pt-8 pb-8 px-6 xl:max-w-360 mx-auto" id="Tests">
        <div
          className="text-3xl font-bold text-center mb-8"
          data-tina-field={tinaField(data, "titleen")}
        >
          {data[`title${language}`]}
        </div>
        <div className="w-full flex flex-wrap justify-center gap-2 sm:gap-3">
          {data.items?.length &&
            data.items.map((item, index) => {
              return <TestsItem data={item} key={index} />;
            })}
        </div>
      </div>
    </div>
  );
};

const defaultTests = {
  namezh: "示例测试",
  nameen: "Sample Testing",
};

export const testsBlockSchema = {
  name: "tests",
  label: "Tests",
  ui: {
    previewSrc: `${basePath}/blocks/updates.png`,
    defaultItem: {
      items: [defaultTests],
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
      type: "object",
      label: "Tests",
      name: "items",
      list: true,
      ui: {
        itemProps: (item) => {
          return {
            label: item?.namezh || item?.nameen,
          };
        },
        defaultItem: {
          ...defaultTests,
        },
      },
      fields: [
        {
          type: "string",
          label: "Chinese Name",
          name: "namezh",
        },
        {
          type: "string",
          label: "English Name",
          name: "nameen",
        },
      ],
    },
  ],
};
