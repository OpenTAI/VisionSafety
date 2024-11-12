import React, { useState, useEffect } from "react";
import Link from "next/link";
import { tinaField } from "tinacms/dist/react";
import { GlobalHeader } from "../../tina/__generated__/types";
import menu from "../../assets/img/menu.png";
import menuWhite from "../../assets/img/menuWhite.png";
import logo from "../../assets/img/logo.png";
import logoWhite from "../../assets/img/logoWhite.png";
import closeIcon from "../../assets/img/closeIcon.png";
import Image from "next/image";
import { Drawer, Collapse } from "antd";

export const Header = ({
  data,
  language,
}: {
  data: GlobalHeader;
  language: string;
  changeLan: Function;
}) => {
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [background, setBackground] = useState("rgba(0,37,99,0)");
  const [textColor, setTextColor] = useState("base-blue");
  const [icon, setIcon] = useState(logo);
  const [menuIcon, setMenuIcon] = useState(menu);

  const { Panel } = Collapse;

  useEffect(() => {
    window.addEventListener("scroll", handleScroll);

    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const onOpen = () => {
    setDrawerOpen(true);
  };

  const onClose = () => {
    setDrawerOpen(false);
  };

  const collapseIcon = (panel) => {
    const arr = ["0", "5"];
    if (arr.indexOf(panel?.panelKey) == -1) {
      if (panel?.isActive) {
        return <div className="w-4 h-4 bg-squareMinus bg-contain" />;
      } else {
        return <div className="w-4 h-4 bg-squarePlus bg-contain" />;
      }
    } else {
      return <div className="w-4 h-4 bg-squareArrow bg-contain" />;
    }
  };

  const handleScroll = () => {
    let scrollTop = document.documentElement.scrollTop;
    if (scrollTop > 64) {
      setBackground("base-blue");
      setTextColor("white");
      setIcon(logoWhite);
      setMenuIcon(menuWhite);
    } else {
      setBackground("white");
      setTextColor("base-blue");
      setIcon(logo);
      setMenuIcon(menu);
    }
  };

  return (
    <div className="fixed w-full z-50">
      <div className={`w-full bg-${background}`}>
        <div
          className={`mx-auto max-w-360 px-10 md:px-28 h-16 flex items-center justify-between text-${textColor}`}
        >
          <div className="flex items-center">
            <Image className="ml-0 md:ml-9 w-28 h-7" src={icon} alt="" />
            <div
              className="font-semibold ml-4 text-xl"
              data-tina-field={tinaField(data, "nameen")}
            >
              {data[`name${language}`]}
            </div>
          </div>
          <div className="hidden w-96 md:flex justify-between">
            {data.nav.map((item, index) => {
              return (
                <div
                  key={index}
                  className="font-semibold text-base cursor-pointer"
                >
                  <Link
                    data-tina-field={tinaField(item, "labelen")}
                    href={`${item.href}`}
                  >
                    {item[`label${language}`]}
                  </Link>
                </div>
              );
            })}
          </div>
          <Image
            className="block md:hidden h-5 w-6"
            src={menuIcon}
            onClick={onOpen}
            alt=""
          />
        </div>
      </div>
      {/* <Container size="custom" className="py-0 relative z-10 max-w-8xl">
        <div className="flex items-center justify-between gap-6">
          <h4 className="select-none text-lg font-bold tracking-tight my-4 transition duration-150 ease-out transform">
            <Link
              href="/"
              className="flex gap-1 items-center whitespace-nowrap tracking-[.002em]"
            >
              <Icon
                tinaField={tinaField(data, "icon")}
                parentColor={data.color}
                data={{
                  name: data.icon.name,
                  color: data.icon.color,
                  style: data.icon.style,
                }}
              />
              <span data-tina-field={tinaField(data, "name")}>{data.name}</span>
            </Link>
          </h4>
          <ul className="flex gap-6 sm:gap-8 lg:gap-10 tracking-[.002em] -mx-4">
            {data.nav &&
              data.nav.map((item, i) => {
                const activeItem =
                  (item.href === ""
                    ? router.asPath === "/"
                    : router.asPath.includes(item.href)) && isClient;
                return (
                  <li
                    key={`${item.label}-${i}`}
                    className={`${
                      activeItem ? activeItemClasses[theme.color] : ""
                    }`}
                  >
                    <Link
                      data-tina-field={tinaField(item, "label")}
                      href={`/${item.href}`}
                      className={`relative select-none	text-base inline-block tracking-wide transition duration-150 ease-out hover:opacity-100 py-8 px-4 ${
                        activeItem ? `` : `opacity-70`
                      }`}
                    >
                      {item.label}
                      {activeItem && (
                        <svg
                          className={`absolute bottom-0 left-1/2 w-[180%] h-full -translate-x-1/2 -z-1 opacity-10 dark:opacity-15 ${
                            activeBackgroundClasses[theme.color]
                          }`}
                          preserveAspectRatio="none"
                          viewBox="0 0 230 230"
                          fill="none"
                          xmlns="http://www.w3.org/2000/svg"
                        >
                          <rect
                            x="230"
                            y="230"
                            width="230"
                            height="230"
                            transform="rotate(-180 230 230)"
                            fill="url(#paint0_radial_1_33)"
                          />
                          <defs>
                            <radialGradient
                              id="paint0_radial_1_33"
                              cx="0"
                              cy="0"
                              r="1"
                              gradientUnits="userSpaceOnUse"
                              gradientTransform="translate(345 230) rotate(90) scale(230 115)"
                            >
                              <stop stopColor="currentColor" />
                              <stop
                                offset="1"
                                stopColor="currentColor"
                                stopOpacity="0"
                              />
                            </radialGradient>
                          </defs>
                        </svg>
                      )}
                    </Link>
                  </li>
                );
              })}
          </ul>
        </div>
        <div
          className={`absolute h-1 bg-gradient-to-r from-transparent ${
            data.color === "primary" ? `via-white` : `via-black dark:via-white`
          } to-transparent bottom-0 left-4 right-4 -z-1 opacity-5`}
        />
      </Container> */}
      <Drawer
        title={null}
        onClose={onClose}
        open={drawerOpen}
        width={414}
        maskClosable={false}
        style={{ background: "#000" }}
        closeIcon={null}
      >
        <div className="flex items-center justify-between mb-7">
          <Image src={logoWhite} className="w-28" alt="" />
          <Image src={closeIcon} className="w-5 h-5" onClick={onClose} alt="" />
        </div>
        <div>
          <Collapse
            bordered={false}
            accordion
            expandIconPosition="end"
            ghost
            expandIcon={(item) => collapseIcon(item)}
          >
            {data.nav.map((item, index) => {
              return (
                <div key={index}>
                  <div className="w-full h-[2px] bg-white"></div>
                  <Panel
                    header={
                      <div>
                        <div className="py-4 text-white font-semibold text-xl">
                          <Link
                            // data-tina-field={tinaField(item, "label")}
                            href={`/${item.href}`}
                          >
                            {item[`label${language}`]}
                          </Link>
                        </div>
                      </div>
                    }
                    key={index}
                  // collapsible={"disabled"}
                  ></Panel>
                </div>
              );
            })}
          </Collapse>
          <div className="w-full h-[2px] bg-white"></div>
        </div>
      </Drawer>
    </div>
  );
};
