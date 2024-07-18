import { Hero } from "../components/blocks/hero";
import Layout from "../components/layout/layout";

export default function FourOhFour() {
  return (
    <Layout>
      <Hero
        data={{
          headline1en: "404 – Page Not Found",
          headline2en: "Oops! It seems there's nothing here, how embarrassing.",
        }}
      />
    </Layout>
  );
}
