class Monitor():
    """
    updateDirPerCategory: 카테고리 별로 방향 라벨링 분포를 확인합니다. 
    updateCount: 생성한 전체 데이터 개수를 셉니다. 
    updataPrompt: 중간중간 생성된 결과물들을 파일로 출력하고 확인할 수 있도록 하는 함수입니다.
    """
    def __init__(self, total_num):
        self.count = 0
        self.total_num = total_num
        self.dir_per_category = {}
        
    def updateDirPerCategory(self, direction):
        if(self.dir_per_category.get(direction)) is None:
            self.dir_per_category[direction] = 0
        self.dir_per_category[direction] = self.dir_per_category[direction] + 1
        
    def updateCount(self):
        self.count = self.count + 1
        if(self.count%10 == 0):
            print(self.count, "/", self.total_num)
            
    def updataPrompt(self, prompt, image, response, dir="../"):
        with open(f'{dir}processing_monitor.txt', 'a', encoding='utf-8') as f:
            f.write("\n <prompt>: \n" + prompt+ "\n---\n")
            f.write("\n <image>: \n" + image+ "\n---\n")
            f.write("\n <response>: \n" + response+ "\n---\n")