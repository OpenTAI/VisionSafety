import React, { useEffect, useState } from "react";
import { InferGetStaticPropsType } from "next";
import { Blocks } from "../components/blocks-renderer";
import { useTina } from "tinacms/dist/react";
import { client } from "../tina/__generated__/client";
import Layout from "../components/layout/layout";

export default function HomePage(
  props: InferGetStaticPropsType<typeof getStaticProps>
) {
  const { data } = useTina(props);
  const [language, setLanguage] = useState("en");
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);
  },[])

  useEffect(() => {
    const lan = navigator.language;
    localStorage.setItem("language", lan);
    setLanguage(lan);
  }, []);

  const changeLan = (lan: string) => {
    setLanguage(lan);
    localStorage.setItem("language", lan);
  };

  return (
    <Layout data={data.global as any} language={language} changeLan={changeLan}>
      <Blocks {...data.page} language={language} />
    </Layout>
  );
}

export const getStaticProps = async ({ params }) => {
  const tinaProps = await client.queries.contentQuery({
    relativePath: `${params.filename}.md`,
  });
  const props = {
    ...tinaProps,
    enableVisualEditing: process.env.VERCEL_ENV === "preview",
  };
  return {
    props: JSON.parse(JSON.stringify(props)) as typeof props,
  };
};

export const getStaticPaths = async () => {
  const pagesListData = await client.queries.pageConnection();
  return {
    paths: pagesListData.data.pageConnection?.edges?.map((page) => ({
      params: { filename: page?.node?._sys.filename },
    })),
    fallback: false,
  };
};
