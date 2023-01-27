USE project_db; -- select database which has tables you want to drop

-- need to consider drop sequence after setting key constraints
DROP TABLE IF EXISTS `item`;
DROP TABLE IF EXISTS `house`;
DROP TABLE IF EXISTS `member`;
DROP TABLE IF EXISTS `house_item`;
DROP TABLE IF EXISTS `house_color`;
DROP TABLE IF EXISTS `member_prefer`;
DROP TABLE IF EXISTS `inference_result`;
