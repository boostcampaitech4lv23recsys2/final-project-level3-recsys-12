CREATE DATABASE IF NOT EXISTS `project_db`; -- choose database you want to create

USE `project_db`; -- select database you want to use

-- create item table
CREATE TABLE IF NOT EXISTS `item` (
	`item_id`	INT(10)	NOT NULL PRIMARY KEY,
	`category`	VARCHAR(100)	NULL,
	`rating`	VARCHAR(100)	NULL,
	`review`	VARCHAR(255)	NULL,
	`price`	VARCHAR(100)	NULL,
	`title`	VARCHAR(100)	NULL,
	`seller`	VARCHAR(100)	NULL,
	`discount_rate`	VARCHAR(100)	NULL,
	`image`	VARCHAR(255)	NULL,
	`available_product`	VARCHAR(100)	NULL,
	`predict_price`	VARCHAR(100)	NULL
);

-- create house table
CREATE TABLE IF NOT EXISTS `house` (
	`house_id`	INT(10)	NOT NULL PRIMARY KEY,
	`space`	VARCHAR(100)	NULL,
	`size`	VARCHAR(100)	NULL,
	`work`	VARCHAR(100)	NULL,
	`category`	VARCHAR(100)	NULL,
	`family`	VARCHAR(100)	NULL,
	`region`	VARCHAR(100)	NULL,
	`style`	VARCHAR(100)	NULL,
	`duration`	VARCHAR(100)	NULL,
	`budget`	VARCHAR(100)	NULL,
	`detail`	VARCHAR(255)	NULL,
	`prefer`	INT(10)	NULL,
	`scrab`	INT(10)	NULL,
	`comment`	INT(10)	NULL,
	`views`	INT(10)	NULL,
	`card_space` VARCHAR(100) 	NULL,
	`card_img_url`	VARCHAR(255)	NULL
);

-- create house_item interaction table
CREATE TABLE IF NOT EXISTS `house_item`(
	`idx` INT(10) NOT NULL AUTO_INCREMENT PRIMARY KEY,
	`house_id` INT(10) NOT NULL,
	`item_id` INT(10) NOT NULL
);

-- create member table
CREATE TABLE IF NOT EXISTS `member` (
	`idx` INT(10) NOT NULL AUTO_INCREMENT PRIMARY KEY,
	`member_email` VARCHAR(255) NOT NULL,
	`house_id`	INT(10)	NOT NULL
);

-- create member_prefer table
CREATE TABLE IF NOT EXISTS `member_prefer` (
	`idx` INT(10) NOT NULL AUTO_INCREMENT PRIMARY KEY,
	`member_email` VARCHAR(255) NOT NULL,
	`item_id` INT(10) NOT NULL
);

-- create inference_result table
CREATE TABLE IF NOT EXISTS `inference_result` (
	`idx`	INT(10) NOT NULL AUTO_INCREMENT PRIMARY KEY,
	`member_email`	VARCHAR(255) NOT NULL,
	`item_id`	INT(10) NOT NULL
);

-- create cluster_item_math table
CREATE TABLE IF NOT EXISTS `cluster_item` (
	`idx`	INT(10)	NOT NULL AUTO_INCREMENT PRIMARY KEY,
	`cluster_id`	INT(10)	NOT NULL,
	`item_id`	INT(10)	NOT NULL
);

-- create card table
CREATE TABLE IF NOT EXISTS `card` (
	`card_id`	INT(10)	NOT NULL PRIMARY KEY,
	`img_space`	VARCHAR(100)	NULL,
	`img_url`	VARCHAR(255)	NULL,
	`house_id`	INT(10)	NULL,
	`is_human`	BINARY(1)	NULL
);

-- create house_color table
CREATE TABLE IF NOT EXISTS `house_color` (
	`house_id`	INT(10)	NOT NULL PRIMARY KEY,
	`main_0`	BINARY(1)	NOT NULL	DEFAULT 0,
	`main_1`	BINARY(1)	NOT NULL	DEFAULT 0,
	`main_2`	BINARY(1)	NOT NULL	DEFAULT 0,
	`main_3`	BINARY(1)	NOT NULL	DEFAULT 0,
	`main_4`	BINARY(1)	NOT NULL	DEFAULT 0,
	`main_5`	BINARY(1)	NOT NULL	DEFAULT 0,
	`main_6`	BINARY(1)	NOT NULL	DEFAULT 0,
	`main_7`	BINARY(1)	NOT NULL	DEFAULT 0,
	`main_8`	BINARY(1)	NOT NULL	DEFAULT 0,
	`main_9`	BINARY(1)	NOT NULL	DEFAULT 0,
	`main_10`	BINARY(1)	NOT NULL	DEFAULT 0,
	`main_11`	BINARY(1)	NOT NULL	DEFAULT 0,
	`main_12`	BINARY(1)	NOT NULL	DEFAULT 0,
	`wall_0`	BINARY(1)	NOT NULL	DEFAULT 0,
	`wall_1`	BINARY(1)	NOT NULL	DEFAULT 0,
	`wall_2`	BINARY(1)	NOT NULL	DEFAULT 0,
	`wall_3`	BINARY(1)	NOT NULL	DEFAULT 0,
	`wall_4`	BINARY(1)	NOT NULL	DEFAULT 0,
	`wall_5`	BINARY(1)	NOT NULL	DEFAULT 0,
	`wall_6`	BINARY(1)	NOT NULL	DEFAULT 0,
	`wall_7`	BINARY(1)	NOT NULL	DEFAULT 0,
	`wall_8`	BINARY(1)	NOT NULL	DEFAULT 0,
	`wall_9`	BINARY(1)	NOT NULL	DEFAULT 0,
	`wall_10`	BINARY(1)	NOT NULL	DEFAULT 0,
	`wall_11`	BINARY(1)	NOT NULL	DEFAULT 0,
	`wall_12`	BINARY(1)	NOT NULL	DEFAULT 0,
	`floor_0`	BINARY(1)	NOT NULL	DEFAULT 0,
	`floor_1`	BINARY(1)	NOT NULL	DEFAULT 0,
	`floor_2`	BINARY(1)	NOT NULL	DEFAULT 0,
	`floor_3`	BINARY(1)	NOT NULL	DEFAULT 0,
	`floor_4`	BINARY(1)	NOT NULL	DEFAULT 0,
	`floor_5`	BINARY(1)	NOT NULL	DEFAULT 0,
	`floor_6`	BINARY(1)	NOT NULL	DEFAULT 0,
	`floor_7`	BINARY(1)	NOT NULL	DEFAULT 0,
	`floor_8`	BINARY(1)	NOT NULL	DEFAULT 0,
	`floor_9`	BINARY(1)	NOT NULL	DEFAULT 0,
	`floor_10`	BINARY(1)	NOT NULL	DEFAULT 0,
	`floor_11`	BINARY(1)	NOT NULL	DEFAULT 0,
	`floor_12`	BINARY(1)	NOT NULL	DEFAULT 0
);